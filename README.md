# CapitalSheetsCollector

CapitalSheets - Financial Statement Classifier

<img width="1237" height="1295" alt="Screenshot (70)" src="https://github.com/user-attachments/assets/244a1dd6-2c89-4c58-b074-e8dab1bcb928" />

# Overview
CapitalSheets is a Django-based web application powered by a Multi-Layer Perceptron (MLP) neural network designed to automatically identify and classify financial tables as Balance Sheets, Income Statements, or Cash Flow Statements. 

## This is WIP project and is not always accurate

This is our teams personal project and with recent breakthroughs in pattern recognition tech (AI) we see more software becoming readliy available.

The application enables users to scan SEC filings (or similar documents), extract tables, and classify them using a pre-trained model. It is intended for local network deployment, allowing multiple users on the same network to access and contribute to a shared session state — facilitating collaborative analysis without data loss.
The underlying model is general-purpose: it accepts tokenized table text and predicts categories, making it reusable for custom applications beyond this project.
Model weights and supporting files will be updated periodically as training data expands.



## Purpose
Our team has found that publicly available financial data tools — both free and paid — often normalize or filter statements to highlight standard KPIs, omitting one-time items, footnotes, or non-standard disclosures. 
CapitalSheets addresses this by preserving the complete, unfiltered structure of principal financial statements. These details can reveal important signals (e.g., unusual charges, restructuring costs, or accounting changes) that warrant deeper investigation.
While no single set of financial statements fully captures a company's health or strategy, they provide a critical starting point for thorough analysis.



## Training Data
The current model was trained on financial tables from a curated dataset spanning over 20 years, covering thousands of individual tables across multiple industries:


| Industry | Company Count |
-----------|----------------
| Technology | 4 |
| E-Commerce | 3 |
| Financial       | 5 |
| Manufacturing   | 2 |
| Retail          | 3 |
| Energy          | 6 |
| Transportation  | 1 |
| Food & Beverage | 1 |

## Future updates will expand industry coverage and dataset size.

## Usage

Deploy locally using Django's development server or a production setup (e.g., Gunicorn + Nginx).
Multiple users on the same network can access the app simultaneously and continue work from shared session state.
Extracted and classified tables can be selectively exported as combined CSVs, save path can be edited but CSVs will be saved to Django directory.

Contributions, bug reports, and suggestions for additional training data are welcome.
