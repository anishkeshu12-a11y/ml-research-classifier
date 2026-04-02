import sys
from concurrent import futures 
import grpc
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1 import health
from grpc_generated import service_pb2_grpc
from predictor import predict

class ClassifierServiceServicer(service_pb2_grpc.ClassifierServiceServicer):
    def ClassifyText(self, request, context):
        text = request.text
        # model predict returns a 1x6 dense matrix of probability or boolean values
        # wait, predictor.py returns x.predict(vectors).todense(), which is a numpy matrix
        import numpy as np
        predictions = np.array(predict(text))[0]
        
        from grpc_generated.service_pb2 import PredictResponse
        # the model outputs are in order: 'Computer Science', 'Physics', 'Mathematics', 
        # 'Statistics', 'Quantitative Biology', 'Quantitative Finance'
        # according to ui.py
        
        return PredictResponse(
            computer_science=bool(predictions[0]),
            physics=bool(predictions[1]),
            mathematics=bool(predictions[2]),
            statistics=bool(predictions[3]),
            quantitative_biology=bool(predictions[4]),
            quantitative_finance=bool(predictions[5])
        )

def serve():
  DEFAULT_PORT = 50055   
  port = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT
  HOST = f'localhost:{port}'

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

  service_pb2_grpc.add_ClassifierServiceServicer_to_server(ClassifierServiceServicer(), server)
  health_pb2_grpc.add_HealthServicer_to_server(health.HealthServicer(), server)

  server.add_insecure_port(HOST)
  print(f"gRPC server started and listening on {HOST}")
  # trigger an eager prediction to cache downloads
  try:
      predict("hello world test")
      print("Predictor downloaded/cached required libraries.")
  except Exception as e:
      print("Predictor cache init failed:", e)
      
  server.start()
  server.wait_for_termination()
  
if __name__ == '__main__':
  serve()
