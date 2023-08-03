# =================================================== ==========================
#  Author: Kunal SK Sukhija
# =============================================================================

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import joblib
from tkinter import *
from tkinter import messagebox

lo=joblib.load("Joblibs\owner.joblib")
lt=joblib.load("Joblibs\Transmission.joblib")
ct=joblib.load("Joblibs\onehot.joblib")
sc=joblib.load("Joblibs\SC.joblib")
regressor=joblib.load("Joblibs\Regressor.joblib")

def f():
    try:
        inp=pd.DataFrame({"name":[e1.get()],
                          "year":[e2.get()],
                          "km_driven":[e3.get()],
                          "fuel":[e4.get()],
                          "seller_type":[e5.get()],
                          "transmission":[e6.get()],
                          "owner":[e7.get()],
                          "mileage":[e8.get()],
                          "engine":[e9.get()],
                          "max_power":[e0.get()],
                          "seats":[e10.get()]})

        inp["owner"]=lo.transform(inp["owner"])
        inp["transmission"]=lt.transform(inp["transmission"])
        inp=ct.transform(inp)
        inp=sc.transform(inp)
        lab=Label(fr,text=f"The Price of the Car is:\n₹ {round(regressor.predict(inp)[0],2)}",bg="black",fg="#A0CE37",font=("MV Boli",35))
        lab.place(relx=0.075,rely=0.2,relheight=0.9,relwidth=0.9)
        but=Button(fr,text="HIDE PRICE",bg="#10696D",fg="#C4F7F9",font=("Casat Cap Bold PERSONAL USE",21),command=lambda:lab.destroy())
        but.place(relx=0.48,rely=0.79)
    except:
        messagebox.showwarning("Error","Please Enter the Correct Values")
    

fr=Tk()
wid=fr.winfo_screenwidth()
hit=fr.winfo_screenheight()
fr.geometry(f"{wid}x{hit}")
fr.configure(bg="black")
l=Label(fr,text="Welcome to The Car Price Predictor!",bg="Black",fg="#12C0F3",font=("Casat Cap Bold PERSONAL USE",37))
l.place(relx=0.13,rely=0.13)
l1=Label(fr,text="Enter Name",bg="black",fg="#F3E912",font=("MV Boli",17))
l2=Label(fr,text="Enter Year",bg="black",fg="#F3E912",font=("MV Boli",17))
l3=Label(fr,text="Enter KiloMeters Driven",bg="black",fg="#F3E912",font=("MV Boli",17))
l4=Label(fr,text="Enter Fuel Type",bg="black",fg="#F3E912",font=("MV Boli",17))
l5=Label(fr,text="Enter Seller Type",bg="black",fg="#F3E912",font=("MV Boli",17))
l6=Label(fr,text="Enter Mode of Transmission",bg="black",fg="#F3E912",font=("MV Boli",17))
l7=Label(fr,text="Enter Owner Status",bg="black",fg="#F3E912",font=("MV Boli",17))
l8=Label(fr,text="Enter Mileage",bg="black",fg="#F3E912",font=("MV Boli",17))
l9=Label(fr,text="Enter Engine Capacity",bg="black",fg="#F3E912",font=("MV Boli",17))
l0=Label(fr,text="Enter Maximum Power",bg="black",fg="#F3E912",font=("MV Boli",17))
l10=Label(fr,text="Enter No. of Seats",bg="black",fg="#F3E912",font=("MV Boli",17))
l1.place(relx=0.09,rely=0.29)
l2.place(relx=0.54,rely=0.29)
l3.place(relx=0.09,rely=0.39)
l4.place(relx=0.54,rely=0.39)
l5.place(relx=0.09,rely=0.49)
l6.place(relx=0.54,rely=0.49)
l7.place(relx=0.09,rely=0.59)
l8.place(relx=0.54,rely=0.59)
l9.place(relx=0.09,rely=0.69)
l0.place(relx=0.54,rely=0.69)
l10.place(relx=0.09,rely=0.79)

e1=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e2=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e3=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e4=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e5=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e6=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e7=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e8=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e9=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e0=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e10=Entry(fr,bg="gray",fg="black",font=("MV Boli",17))
e1.place(relx=0.3,rely=0.29,relwidth=0.15)
e2.place(relx=0.78,rely=0.29,relwidth=0.15)
e3.place(relx=0.3,rely=0.39,relwidth=0.15)
e4.place(relx=0.78,rely=0.39,relwidth=0.15)
e5.place(relx=0.3,rely=0.49,relwidth=0.15)
e6.place(relx=0.78,rely=0.49,relwidth=0.15)
e7.place(relx=0.3,rely=0.59,relwidth=0.15)
e8.place(relx=0.78,rely=0.59,relwidth=0.15)
e9.place(relx=0.3,rely=0.69,relwidth=0.15)
e0.place(relx=0.78,rely=0.69,relwidth=0.15)
e10.place(relx=0.3,rely=0.79,relwidth=0.15)

b=Button(fr,text="SHOW PRICE",bg="#10696D",fg="#C4F7F9",font=("Casat Cap Bold PERSONAL USE",20),command=f)
b.place(relx=0.7,rely=0.79)
fr.mainloop()

