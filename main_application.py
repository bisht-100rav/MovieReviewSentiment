import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
from tensorflow import keras


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        data = keras.datasets.imdb
        word_index = data.get_word_index()
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"] = 0
        word_index["<START>"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3

        def review_encode(s):
            encoded = [1]

            for word in s:
                if word.lower() in word_index:
                    encoded.append(word_index[word.lower()])
                else:
                    encoded.append(2)
            return encoded

        model = keras.models.load_model("Moview_review_Model_1.h5")

        #Label
        l1 = tk.Label(root, text='Movie Review Analysis',
                    font=('Arial Bold', 10))
        l1.pack(side='top')

        S = tk.Scrollbar(root)
        T = tk.Text(root)

        # Get Text from the text Box
        def get_text():
            inputValue = T.get("1.0", "end-1c")
            return inputValue

        # Commancd for Button 1
        def on_click():
            line = get_text()
            nline = line.replace(",", "").replace(".", "").replace("(", "").replace(
                ")", "").replace(":", "").replace("\"", "").strip().split(" ")
            encode = review_encode(nline)
            encode = keras.preprocessing.sequence.pad_sequences(
                [encode], value=word_index["<PAD>"], padding="post", maxlen=250)
            predict = model.predict(encode)
            #print(line)
            #print(encode)
            result = 100 * predict[0]
            if result <= 10:
                messagebox.showinfo("Rating", "POOR")
                print(result)
            if result > 10 and result <= 50:
                messagebox.showinfo("Rating", "AVERAGE")
                print(result)
            if result > 50 and result <= 70:
                messagebox.showinfo("Rating", "GOOD")
                print(result)
            if result > 70:
                messagebox.showinfo("Rating", "VERY GOOD")
                print(result)

        # Button 1
        bt = tk.Button(root, text="Analyse", command=on_click)
        bt.pack(side='bottom')

        S.pack(side=tk.RIGHT, fill=tk.Y)
        T.pack(side=tk.LEFT, fill=tk.Y)

        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)

        entry = tk.Entry(root, textvariable=T)


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)
    root.geometry("600x400")
    root.title("SENTIMENT ANALYSIS")
    root.mainloop()
