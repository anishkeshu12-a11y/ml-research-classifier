import tkinter
import customtkinter
from predictor import predict
import numpy

customtkinter.set_appearance_mode('System')
customtkinter.set_default_color_theme('blue')

app = customtkinter.CTk()
app.geometry('800x440')
app.title('Research Paper Classifier')
label = customtkinter.CTkLabel(app, text='Paste Paper Abstract or Content')
label.pack()
textbox = customtkinter.CTkTextbox(app, width=700, height=270, corner_radius=20)
textbox.pack(padx=30, pady=10)
def button_event():
    global textbox
    global buttons
    text = textbox.get('1.0', 'end-1c')
    predictions = numpy.array(predict(text))[0]
    for pred in range(len(predictions)):
        if predictions[pred]:
            buttons[pred].configure(fg_color='blue', text_color_disabled='white')

    
frame1 = customtkinter.CTkFrame(app)   
frame2 = customtkinter.CTkFrame(app)   
buttons = []
for subject in ['Computer Science', 'Physics',	'Mathematics']:
    buttons.append(customtkinter.CTkButton(frame1, text=subject, fg_color='gray', corner_radius=4, state='disabled', text_color_disabled='black'))
for subject in ['Statistics', 'Quantitative Biology', 'Quantitative Finance']:
    buttons.append(customtkinter.CTkButton(frame2, text=subject, fg_color='gray', corner_radius=4, state='disabled', text_color_disabled='black'))
button = customtkinter.CTkButton(master=app, text="Classify", command=button_event, anchor=tkinter.CENTER)

button.pack(padx=20, pady=10)

for btn in buttons[:3]:
    btn.pack(padx=5, side=tkinter.LEFT)
frame1.pack()
for btn in buttons[3:]:
    btn.pack(padx=5, pady=5, side=tkinter.LEFT)
frame2.pack()
app.mainloop()