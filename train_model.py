import json
import pathlib
import pickle

from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing

from create_model import SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION, OUTPUT_DIR, load_data


sales_demographic_data, target = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)


x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
    x, y, random_state=42)

model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                neighbors.KNeighborsRegressor()).fit(
                                    x_train, y_train)

output_dir = pathlib.Path(OUTPUT_DIR)
output_dir.mkdir(exist_ok=True)

# Output model artifacts: pickled model and JSON list of features
pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
json.dump(list(x_train.columns),
            open(output_dir / "model_features.json", 'w'))