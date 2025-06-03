import os
import csv
from Color_Distribution_method import analyze_color_distribution

def process_folder(folder_path, label):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_path, filename)
            try:
                prob = analyze_color_distribution(path)
                results.append((filename, label, prob))
            except Exception as e:
                print(f"❌ Ошибка в {filename}: {e}")
    return results

def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        return {(row['filename'], row['label']): float(row['prob_fake']) for row in reader}

def main():
    folders = {
        'real_phone': 'real_phone',
        'real_net': 'real_net',
        'fake': 'fake'
    }
    output_file = 'results_color.csv'

    existing_results = load_existing_results(output_file)
    all_results = []

    for label, path in folders.items():
        all_results.extend(process_folder(path, label))

    for r in all_results:
        existing_results[(r[0], r[1])] = r[2]

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'prob_fake'])
        for (filename, label), prob in sorted(existing_results.items()):
            writer.writerow([filename, label, f"{prob:.4f}"])

    print(f"✅ Результаты сохранены в {output_file}")

if __name__ == '__main__':
    main()
