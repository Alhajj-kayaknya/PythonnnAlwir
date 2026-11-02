# ==========================================
# CORE ENGINE: AI SPAM FILTER (NAIVE BAYES)
# ==========================================

# Data Fakta di Database Email Lu
# Settingan Laplace Smoothing (Bansos AI)
alpha = 1 # Angka bansos atas
vocab_size = 50 # Total kata unik di database lu

# Kodingan lu yang kemaren di-upgrade dikit:
# Misal ngitung probabilitas kata 'Skibidi' di email Normal
p_skibidi_kalo_normal = (0 + alpha) / (200 + vocab_size)

p_spam = 0.3
p_normal = 0.7

p_hadiah_kalo_spam = 0.6
p_hadiah_kalo_normal = 0.1

# Step 1: Aturan Probabilitas Total (Nyari Total Evidence)
jalur_spam = p_spam * p_hadiah_kalo_spam
jalur_normal = p_normal * p_hadiah_kalo_normal

total_hadiah = jalur_spam + jalur_normal

# Step 2: Teorema Bayes Mutlak (Update Keyakinan AI)
p_spam_kalo_ada_hadiah = jalur_spam / total_hadiah

# Tampilkan ke terminal (Pake f-string biar rapi)
print("=== HASIL ANALISIS AI ===")
print(f"Tebakan awal AI (tanpa bukti)     : {p_spam * 100}%")
print(f"Total bukti kata 'Hadiah' beredar : {total_hadiah * 100}%")
print(f"Tebakan AI setelah nemu 'Hadiah'  : {p_spam_kalo_ada_hadiah * 100:.0f}%\n")

if p_spam_kalo_ada_hadiah > 0.5:
    print("STATUS: EMAIL INI MUTLAK SPAM! BUANG KE TRASH! 🗑️")
else:
    print("STATUS: EMAIL AMAN DAWGG 🛡️")


# Simulasi nemu kata ghaib
kata_test = "Mulyono"
kemunculan = 0
total_kata_normal = 200
vocab_size = 50

print("\n=== SYSTEM SCANNING KATA GHAIB: MULYONO ===")

# 1. Kalo AI lu bego (Tanpa Laplace)
prob_kiamat = kemunculan / total_kata_normal
print(f"Probabilitas Tanpa Laplace : {prob_kiamat}")
# Terminal bakal nampilin: 0.0 (Bakal bikin error kiamat pas dikali)

# 2. Kalo AI lu pake cheat code (Pake Laplace)
prob_selamat = (kemunculan + 1) / (total_kata_normal + vocab_size)
print(f"Probabilitas Pake Laplace  : {prob_selamat}")
# Terminal bakal nampilin: 0.004

peluang = 0.1
hasil = peluang * prob_selamat
print(f"Hasil Ketika Dikaliin : {hasil}")