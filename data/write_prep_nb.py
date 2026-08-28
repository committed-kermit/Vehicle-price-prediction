import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preprocessing Phase\n",
    "This notebook demonstrates the preprocessing of the vehicle dataset, replicating the creation of the preprocessed train/test datasets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.preprocessing import OneHotEncoder"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load the raw datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_raw = joblib.load('X_train.pkl')\n",
    "X_test_raw = joblib.load('X_test.pkl')\n",
    "y_train = joblib.load('y_train.pkl')\n",
    "y_test = joblib.load('y_test.pkl')\n",
    "\n",
    "print(\"X_train shape:\", X_train_raw.shape)\n",
    "print(\"X_test shape:\", X_test_raw.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Preprocessing Steps\n",
    "1. Apply log transformation (`log1p`) to the highly skewed `km_driven` variable.\n",
    "2. Rename the `name` column to `model` to represent the vehicle name.\n",
    "3. One-hot encode the categorical variables: `fuel`, `seller_type`, `transmission`, and `owner` using `ColumnTransformer`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_dataset(df_raw, preprocessor=None, fit=False):\n",
    "    df = df_raw.copy()\n",
    "    # Log transform km_driven\n",
    "    df['km_driven'] = np.log1p(df['km_driven'])\n",
    "    # Rename name to model\n",
    "    df = df.rename(columns={'name': 'model'})\n",
    "    \n",
    "    cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']\n",
    "    \n",
    "    if fit:\n",
    "        preprocessor = ColumnTransformer(\n",
    "            transformers=[\n",
    "                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols)\n",
    "            ], remainder='passthrough')\n",
    "        transformed_data = preprocessor.fit_transform(df)\n",
    "    else:\n",
    "        transformed_data = preprocessor.transform(df)\n",
    "        \n",
    "    cols = preprocessor.get_feature_names_out()\n",
    "    # Clean feature names (remove remainder__ prefix)\n",
    "    cols = [col.replace('remainder__', '') for col in cols]\n",
    "    \n",
    "    df_processed = pd.DataFrame(transformed_data, columns=cols, index=df.index)\n",
    "    \n",
    "    # Cast columns back to original/correct datatypes\n",
    "    for col in df_processed.columns:\n",
    "        if col not in ['model', 'year', 'km_driven']:\n",
    "            df_processed[col] = df_processed[col].astype(float)\n",
    "        elif col in ['year']:\n",
    "            df_processed[col] = df_processed[col].astype(int)\n",
    "        elif col in ['km_driven']:\n",
    "            df_processed[col] = df_processed[col].astype(float)\n",
    "        elif col in ['model']:\n",
    "            df_processed[col] = df_processed[col].astype(str)\n",
    "            \n",
    "    return df_processed, preprocessor"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Fit and Transform"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_prep, preprocessor = preprocess_dataset(X_train_raw, fit=True)\n",
    "X_test_prep, _ = preprocess_dataset(X_test_raw, preprocessor=preprocessor, fit=False)\n",
    "\n",
    "print(\"Preprocessed X_train shape:\", X_train_prep.shape)\n",
    "print(\"Preprocessed X_test shape:\", X_test_prep.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Verify and save the preprocessed outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the preprocessed datasets into the models folder\n",
    "joblib.dump(X_train_prep, '../models/X_train.pkl')\n",
    "joblib.dump(X_test_prep, '../models/X_test.pkl')\n",
    "joblib.dump(y_train, '../models/y_train.pkl')\n",
    "joblib.dump(y_test, '../models/y_test.pkl')\n",
    "\n",
    "print(\"Preprocessing completed and pickled files saved to ../models/.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('data/preprocessing.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
