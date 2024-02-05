#include "utility.h"

int main() {
    // Open the file
    ifstream file("input_data/NVDA.csv");
    if (!file.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }

    vector<tm> dates;
    vector<float> values;
	dates.reserve(SERIES_LENGHT);
	values.reserve(SERIES_LENGHT);

    string line;

	// Skip first line (because it contains the header row)
	getline(file, line);

    while (getline(file, line)) {
        istringstream line_stream(line);
        string token;
        vector<string> tokens;
		tokens.reserve(7 * SERIES_LENGHT);

        while (getline(line_stream, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() >= 6) {
			// Get date and close value
            tm date = {};
            istringstream date_stream(tokens[0]);
            date_stream >> get_time(&date, "%Y-%m-%d");
            dates.push_back(date);
            values.push_back(stof(tokens[4])); 
        }
    }

    // Stampa le date e i valori
    cout << "Dates:" << endl;
    for (const auto& date : dates) {
        cout << put_time(&date, "%Y-%m-%d") << endl;
    }

    cout << "Values:" << endl;
    for (const auto& value : values) {
        cout << value << endl;
    }

    // Chiudi il file
    file.close();

    return 0;
}
