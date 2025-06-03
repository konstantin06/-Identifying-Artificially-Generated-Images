import os
import csv
from The_entropy_complexity_method import analyze_entropy_complexity

def process_folder(folder_path, label):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            try:
                prob = analyze_entropy_complexity(image_path, show_images=False)  # отключено
                results.append((filename, label, prob))
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    return results

def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return {}

    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        return {(row['filename'], row['label']): float(row['prob_fake']) for row in reader}

def main():
    real_phone_dir = 'real_phone'
    real_net_dir = 'real_net'
    fake_dir = 'fake'
    output_file = 'results_entropy.csv'

    # Загружаем существующие строки (если есть)
    existing_results = load_existing_results(output_file)

    # Обновляем/добавляем новые строки
    new_results = process_folder(real_phone_dir, 'real_phone') + process_folder(real_net_dir, 'real_net') + process_folder(fake_dir, 'fake')
    for entry in new_results:
        key = (entry[0], entry[1])
        existing_results[key] = entry[2]

    # Сохраняем результат
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'label', 'prob_fake'])
        for (filename, label), prob_fake in sorted(existing_results.items()):
            writer.writerow([filename, label, f"{prob_fake:.4f}"])

    print(f"\n✅ Results saved/updated to {output_file}")

if __name__ == '__main__':
    main()
