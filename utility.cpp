#include "utility.h"


float calculateMeanInRange(const vector<float>& values, int start, int end) {
    float sum = 0.0f;
    for (int i = start; i <= end; ++i) {
        sum += values[i];
    }
    return sum / (end - start);
}

float calculateStandardDeviationInRange(const vector<float>& values, int start, int end) {
    float mean = calculateMeanInRange(values, start, end);
    float variance = 0.0f;
    for (int i = start; i <= end; ++i) {
        variance += pow(values[i] - mean, 2);
    }
    variance /= (end - start);
    return sqrt(variance);
}

float calculateCorrelationCoefficient(const vector<float>& values1, const vector<float>& values2) {
    if (values1.size() != values2.size()) {
        cerr << "Error: The two vectors must have the same length." << endl;
        return 0.0f;
    }

    float mean1 = calculateMeanInRange(values1, 0, values1.size());
    float mean2 = calculateMeanInRange(values2, 0, values2.size());
    float stdDev1 = calculateStandardDeviationInRange(values1, 0, values1.size());
    float stdDev2 = calculateStandardDeviationInRange(values2, 0, values2.size());

    float sum = 0.0f;
    for (size_t i = 0; i < values1.size(); ++i) {
        sum += (values1[i] - mean1) * (values2[i] - mean2);
    }

    return sum / (values1.size() * stdDev1 * stdDev2);
}
