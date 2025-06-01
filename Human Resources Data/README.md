# HR Analytics: Employee Attrition Prediction

This project uses an HR dataset to predict **employee attrition** (i.e., whether an employee is likely to leave the company). The dataset includes a wide range of employee-related features such as demographics, job satisfaction, salary, performance, and work-life balance metrics.

---

## 📁 Dataset Overview

The dataset contains structured HR data with the following features:

| Feature                     | Description |
|----------------------------|-------------|
| `AGE`                      | Numerical value representing the employee's age. |
| `ATTRITION`                | Target variable: 0 = No (employee stayed), 1 = Yes (employee left). |
| `BUSINESS TRAVEL`          | 1 = No Travel, 2 = Travel Frequently, 3 = Travel Rarely. |
| `DAILY RATE`               | Numerical value representing salary level. |
| `DEPARTMENT`               | 1 = HR, 2 = R&D, 3 = Sales. |
| `DISTANCE FROM HOME`       | Numerical value representing the distance between home and workplace. |
| `EDUCATION`                | Numerical score of educational attainment. |
| `EDUCATION FIELD`          | 1 = HR, 2 = Life Sciences, 3 = Marketing, 4 = Medical Sciences, 5 = Others, 6 = Technical. |
| `EMPLOYEE COUNT`           | Numerical value (constant for all rows, can be dropped). |
| `EMPLOYEE NUMBER`          | Unique employee ID. |
| `ENVIROMENT SATISFACTION`  | Numerical score for satisfaction with the work environment. |
| `GENDER`                   | 1 = Female, 2 = Male. |
| `HOURLY RATE`              | Hourly salary rate. |
| `JOB INVOLVEMENT`          | Numerical score for employee job involvement. |
| `JOB LEVEL`                | Numerical job level within the organization. |
| `JOB ROLE`                 | 1 = HC Rep, 2 = HR, 3 = Lab Technician, 4 = Manager, 5 = Managing Director, 6 = Research Director, 7 = Research Scientist, 8 = Sales Executive, 9 = Sales Representative. |
| `JOB SATISFACTION`         | Numerical job satisfaction score. |
| `MARITAL STATUS`           | 1 = Divorced, 2 = Married, 3 = Single. |
| `MONTHLY INCOME`           | Monthly salary. |
| `MONTHY RATE`              | Monthly rate (possibly salary or cost center allocation). |
| `NUMCOMPANIES WORKED`      | Number of companies the employee has previously worked at. |
| `OVER 18`                  | 1 = Yes, 2 = No (all values should be 1 — can be dropped). |
| `OVERTIME`                 | 1 = No, 2 = Yes. |
| `PERCENT SALARY HIKE`      | Percentage increase in salary from the previous year. |
| `PERFORMANCE RATING`       | Performance rating score. |
| `RELATIONS SATISFACTION`   | Score of satisfaction with interpersonal relations. |
| `STANDARD HOURS`           | Number of standard working hours (usually constant — can be dropped). |
| `STOCK OPTIONS LEVEL`      | Level of stock options awarded. |
| `TOTAL WORKING YEARS`      | Total years of professional experience. |
| `TRAINING TIMES LAST YEAR` | Number of hours spent in training over the last year. |
| `WORK LIFE BALANCE`        | Rating of work-life balance. |
| `YEARS AT COMPANY`         | Total number of years spent at the current company. |
| `YEARS IN CURRENT ROLE`    | Years spent in the current role. |
| `YEARS SINCE LAST PROMOTION` | Years since the last promotion. |
| `YEARS WITH CURRENT MANAGER` | Years spent under the current manager. |

---

## 🎯 Objective

To build a machine learning model that can predict whether an employee will leave the company (`Attrition = 1`) based on the available features.

---

## 🛠️ Potential Use Cases

- Employee retention analysis.
- Predictive workforce planning.
- Targeted engagement and satisfaction programs.

---

## 📌 Notes

- Some features such as `Employee Count`, `Standard Hours`, and `Over 18` might be constant and can be removed during preprocessing.
- Categorical variables are encoded numerically and may need one-hot encoding or label decoding during exploration.

---

## 📂 File

Ensure that your CSV file is named and located as:

```bash
./Human Resources Data/Human_Resources.csv
