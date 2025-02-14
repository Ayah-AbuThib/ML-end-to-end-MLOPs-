# Diamond Price Prediction

## 📌 Project Overview

This project focuses on predicting diamond prices using machine learning techniques. The dataset contains information on 54,000 diamonds, including their price and various physical attributes. It serves as an excellent resource for learning data analysis and visualization.

## 📊 Dataset Information

- **Price:** Diamond price in US dollars ($326 - $18,823)
- **Carat:** Weight of the diamond (0.2 - 5.01)
- **Cut:** Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **Color:** Diamond color, from J (worst) to D (best)
- **Clarity:** Measurement of how clear the diamond is (I1 (worst) to IF (best))
- **x:** Length in mm (0 - 10.74)
- **y:** Width in mm (0 - 58.9)
- **z:** Depth in mm (0 - 31.8)
- **Depth:** Total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43 - 79)
- **Table:** Width of the top of the diamond relative to the widest point (43 - 95)

## 📂 Project Structure

```
├── default_repo
│   ├── .ssh_tunnel
│   ├── charts
│   ├── custom
│   │   ├── api.py
│   │   ├── colorful_destiny.py
│   │   ├── inference.py
│   │   ├── limitless_monk.py
│   │   ├── prismatic_field.py
│   │   ├── symmetrical_forest.py
│   ├── data_exporters
│   │   ├── build.py
│   ├── data_loaders
│   │   ├── load_data.py
│   │   ├── train.csv
│   ├── dbt
│   ├── extensions
│   ├── interactions
│   ├── interface
│   │   ├── interface.py
│   ├── markdowns
│   ├── pipelines
│   │   ├── data_preprocessing
│   │   │   ├── metadata.yaml
│   │   │   ├── triggers.yaml
│   ├── predict
│   │   ├── metadata.yaml
│   ├── scratchpads
│   ├── transformers
│   │   ├── data_preprocessing.py
│   │   ├── fill_in_missing_values.py
│   │   ├── kinetic_healer.py
│   │   ├── model.py
│   │   ├── random_forest_(tuned).joblib
│   ├── utils
├── .gitignore
├── global_data_products.yaml
├── io_config.yaml
```

## 🚀 How to Use

1. Clone the repository:
    
    ```sh
    git clone https://github.com/your_username/diamond-price-prediction.git
    ```
    
2. Install dependencies:
    
    ```sh
    pip install -r requirements.txt
    ```
    
3. Run data preprocessing:
    
    ```sh
    python default_repo/transformers/data_preprocessing.py
    ```
    
4. Train the model:
    
    ```sh
    python default_repo/transformers/modeling.py
    ```
    
5. Run predictions:
    
    ```sh
    python default_repo/custom/inference.py
    ```
