from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# anggap aja 1 = spam, 0 = normal
faktaLapangan = [1, 1, 1, 0, 0, 0, 0] # [TN, FP] = [4 0]
tebakanAI = [1, 1, 0, 0, 0, 0, 0]     # [FN, TP] = [1 2]

matriks = confusion_matrix(faktaLapangan, tebakanAI)
akurasiAI = accuracy_score(faktaLapangan, tebakanAI)
presisiAI = precision_score(faktaLapangan, tebakanAI)
recallAI = recall_score(faktaLapangan, tebakanAI)

print("=== LAPORAN CONFUSION MATRIKS ===")
print(matriks)
print(f"Akurasi : {akurasiAI * 100:.2f}%")
print(f"Presisi : {presisiAI * 100:.2f}%")
print(f"Recall : {recallAI * 100:.2f}%")