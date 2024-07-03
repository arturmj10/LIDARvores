import tkinter as tk
from tkinter import filedialog

def selecionar_arquivo():
    caminho_arquivo = filedialog.askopenfilename()
    return caminho_arquivo

def confirmar():
    global caminho_arquivo, root
    print("Arquivo selecionado:", caminho_arquivo.get())
    root.destroy()

def cancelar():
    global root
    root.destroy()

def criar_interface():
    global caminho_arquivo, root
    root = tk.Tk()
    root.title("Selecionar Arquivo")
    root.geometry("500x300")  

    caminho_arquivo = tk.StringVar()
    caminho_arquivo.set("Nenhum arquivo selecionado")

    def on_select():
        caminho = selecionar_arquivo()
        if caminho:
            caminho_arquivo.set(caminho)

    
    botao_selecionar = tk.Button(root, text="Selecionar Arquivo", command=on_select, padx=10, pady=5)
    botao_selecionar.pack(pady=20)

    
    label_arquivo = tk.Label(root, textvariable=caminho_arquivo, wraplength=480, justify='center')
    label_arquivo.pack(pady=20)

    
    botao_confirmar = tk.Button(root, text="Confirmar", command=confirmar, padx=10, pady=5, bg='#4CAF50', fg='white')
    botao_confirmar.pack(side=tk.LEFT, padx=10, pady=10)

    
    botao_cancelar = tk.Button(root, text="Cancelar", command=cancelar, padx=10, pady=5, bg='#FF5733', fg='white')
    botao_cancelar.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()

    return caminho_arquivo.get()
