import os
import csv
from Fourier_Transform_method import analyze_fourier

def process_folder(folder_path, label):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            try:
                prob = analyze_fourier(image_path)
                results.append((filename, label, prob))
            except Exception as e:
                print(f"❌ Ошибка при обработке {filename}: {e}")
    return results

def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        return {(row['filename'], row['label']): float(row['prob_fake']) for row in reader}

def main():
    folders = {
        'real_phone': 'real_phone',
        'real_net': 'real_net',
        'fake': 'fake'
    }
    output_file = 'results_fourier.csv'

    existing_results = load_existing_results(output_file)

    all_results = []
    for label, path in folders.items():
        all_results.extend(process_folder(path, label))

    # Обновляем или добавляем записи
    for entry in all_results:
        key = (entry[0], entry[1])
        existing_results[key] = entry[2]

    # Сохраняем CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'prob_fake'])
        for (filename, label), prob_fake in sorted(existing_results.items()):
            writer.writerow([filename, label, f"{prob_fake:.4f}"])

    print(f"\n✅ Результаты сохранены в {output_file}")

if __name__ == '__main__':
    main()
