import base64
import binascii
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
from time import perf_counter

# Fungsi untuk memperbaiki padding base64
def fix_base64_padding(s):
    return s + '=' * (-len(s) % 4)

# Inisialisasi kunci AES yang sama
key = b'Enkripsiplat3652'  # Kunci 16 byte untuk AES-128

# Buka file yang berisi teks terenkripsi
with open("car_plate_data.txt", "r") as file:
    lines = file.readlines()

# Siapkan file output untuk menyimpan hasil
with open("decrypted_car_plate_data.txt", "w") as output_file:
    # Tulis header dengan lebar kolom yang lebih besar
    output_file.write("{:<15}\t{:<15}\t{:<10}\t{:<50}\t{:<50}\t{:<20}\n".format("Tanggal", "Waktu", "NoPlat", "Enkripsi", "Decode", "Waktu Dekripsi (ms)"))
    
    # Proses setiap baris dalam file untuk mendekripsi teks
    for line in lines:
        # Pisahkan bagian-bagian dari baris yang sesuai dengan format yang telah disimpan
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue  # Lewati baris yang tidak memiliki cukup bagian
        
        no_plat = parts[0]  # Nomor plat
        tanggal = parts[1]  # Tanggal
        waktu = parts[2]    # Waktu
        encrypted_text_base64 = parts[4]  # Teks terenkripsi dalam base64
        
        # Perbaiki padding base64 jika diperlukan
        encrypted_text_base64 = fix_base64_padding(encrypted_text_base64)
        
        # Decode base64 untuk mendapatkan teks terenkripsi asli
        try:
            encrypted_text = base64.b64decode(encrypted_text_base64)
        except binascii.Error as e:
            print(f"Error decoding base64: {e}")
            continue
        
        # Inisialisasi cipher dengan kunci yang sama dalam mode ECB
        cipher = AES.new(key, AES.MODE_ECB)
        
        # Mengukur waktu dekripsi AES dalam milidetik menggunakan perf_counter
        start_time = perf_counter()
        
        try:
            # Dekripsi teks dan hapus padding
            decrypted_text = unpad(cipher.decrypt(encrypted_text), AES.block_size).decode('utf-8')
            end_time = perf_counter()
            decryption_time = (end_time - start_time) * 1000  # Konversi ke milidetik

            # Simpan hasil tanggal, waktu, enkripsi, dekripsi, dan waktu dekripsi ke dalam file output
            output_file.write("{:<15}\t{:<15}\t{:<10}\t{:<50}\t{:<50}\t{:<20.3f}\n".format(tanggal, waktu, no_plat, encrypted_text_base64, decrypted_text, decryption_time))
        except ValueError as e:
            print(f"Error decrypting text: {e}")
            continue
