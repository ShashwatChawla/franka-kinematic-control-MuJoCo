import csv
import matplotlib.pyplot as plt

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader) 
        for row in csv_reader:
            data.append((float(row[0]), float(row[1])))
    return data

def plot_data(data):
    x_values, y_values = zip(*data)
    plt.plot(x_values, y_values,linestyle='-')
    plt.axhline(y=15, color='r', linestyle='--', label='y=15')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.title('Part(2) Impedance v/s Time')
    plt.grid(True)
    plt.show()

def plot_data(force_, impedance_):
    x_values_f, y_values_f = zip(*force_)
    x_values_im, y_values_img = zip(*impedance_)
    plt.plot(x_values_f, y_values_f,linestyle='-', label='Force', color='blue')
    plt.plot(x_values_im, y_values_img,linestyle='--', label='Impedance', color='red')
    plt.axhline(y=15, color='r', linestyle='--', label='y=15')
    plt.xlabel('Time')
    plt.ylabel('Force')
    plt.legend()
    plt.title('Part(2) Force-Impedance v/s Time Plot')
    plt.grid(True)  
    plt.show()

if __name__ == "__main__":
    # Specify the path to your CSV file
    force_file_path = "force_vs_time.csv"
    impedance_file_path = "impedance_vs_time.csv"
    # Read data from CSV file
    data_from_force  = read_csv(force_file_path)
    data_from_impedance = read_csv(impedance_file_path)
    # Plot the data
    plot_data(data_from_force, data_from_impedance)
