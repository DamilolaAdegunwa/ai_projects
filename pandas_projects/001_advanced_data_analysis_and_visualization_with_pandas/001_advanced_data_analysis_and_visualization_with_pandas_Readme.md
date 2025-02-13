# Employee Data Analysis Project

## Overview
This project provides a comprehensive analysis of an employee dataset using Python and various data analysis libraries. It includes functions to evaluate performance, salary distribution, hiring trends, tenure, and correlations between key employee attributes.

## Features
The project includes the following analyses:

1. **Department Performance Analysis** - Computes the average performance score for each department.
2. **Salary Pivot Table** - Generates a pivot table to analyze salary distribution across cities and departments.
3. **Hiring Trend Analysis** - Performs a time-series analysis of hiring trends from 2010 to 2023.
4. **Average Salary by Age Group** - Categorizes employees by age groups and computes their average salary.
5. **Top Cities by Performance** - Identifies the top 5 cities with the highest average performance scores.
6. **Salary Distribution by Age** - Uses boxplots to visualize salary distribution by age.
7. **Longest Tenure by Department** - Determines which department has employees with the longest average tenure.
8. **Performance-Salary Correlation** - Analyzes the correlation between employee performance scores and salaries.
9. **Department Size Visualization** - Displays the proportion of employees in each department.
10. **Hiring Trends by City** - Examines hiring trends across different cities over the years.

## Additional Analyses (Commented in the Code)
- Top 5 highest-paid employees in each department
- Percentage of employees in each department with high performance scores
- Correlation analysis between numerical features
- Average tenure of employees by department and gender
- Anomaly detection in salary distribution using Z-score analysis

## Technologies Used
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Scipy**: Statistical analysis for anomaly detection

## Installation
To run this project locally, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/employee-analysis.git
   cd employee-analysis
   ```
2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate  # For Windows
   ```
3. **Install Required Libraries**
   ```bash
   pip install pandas matplotlib seaborn openpyxl scipy
   ```

## Usage
1. **Ensure the Dataset is Available**
   - Place `employee_dataset.xlsx` in the project directory.
2. **Run the Analysis**
   ```bash
   python employee_analysis.py
   ```
3. **Review the Output**
   - Printed summaries in the console.
   - Visual plots for trend analysis and distributions.

## File Structure
```
employee-analysis/
│-- employee_analysis.py   # Main script with analysis functions
│-- employee_dataset.xlsx  # Sample dataset
│-- README.md              # Project documentation
│-- requirements.txt       # List of required dependencies
```

## Example Output
```bash
Average Performance Scores by Department:
   Department  Avg_Performance_Score
0  Marketing                   4.2
1       HR                      3.8
...

Correlation between Performance Score and Salary: 0.56
```

## Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
