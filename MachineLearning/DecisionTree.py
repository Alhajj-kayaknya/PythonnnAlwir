from sklearn.tree import DecisionTreeClassifier, export_text

faktaLapangan = [
    [1, 1, 0], # kolom 1 diskon, kolom 2 link aneh, kolom 3 domain sekolah
    [1, 0, 0], # 1 = ada, 0 = ga ada
    [0, 0, 1],
    [0, 0, 0]
]

KeputusanAkhir = ["SPAM (Leaf 1)", "NORMAL (Leaf 2)", "PENTING (Leaf 3)", "NORMAL (Leaf 4)"]

MesinDecision = DecisionTreeClassifier()
MesinDecision.fit(faktaLapangan, KeputusanAkhir)

Visual = export_text(MesinDecision, feature_names=["Kata Diskon", "Link_aneh", "Domain_sekolah"])
print("=== BENTUK DECISION TREE ===")
print(Visual)

emailBaru = [[0, 0, 1]]
TebakanAI = MesinDecision.predict(emailBaru)

print(f"\nTarget : email dari wali kelas (smkn1.sch.id)")
print(f"HASIL TEBAKAN AI : {TebakanAI[0]}")

pSpam = 2/4
pPenting = 2/4

giniNodeB = 1 - (pSpam**2) - (pPenting**2)
print(f"Nilai Gini Impurity Node B : {giniNodeB}")
