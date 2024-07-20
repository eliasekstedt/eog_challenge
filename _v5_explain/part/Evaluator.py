
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

class Evaluator:
    def __init__(self, runpath, foldpath, model, loader, device):
        pred_data = self.evaluate(model, loader, device)
        eval_data = self.assemble_eval_data(foldpath, pred_data)
        eval_data.to_csv(f'{runpath}eval_data.csv', index=False)
        self.cmatrix = self.get_cmatrix(eval_data)
        from util.tools_plotting import plot_cmatrix
        plot_cmatrix(self.cmatrix, runpath)
        self.score = self.get_score(eval_data)

    def evaluate(self, model, loader, device):
        model.eval()
        with torch.no_grad():
            row_id, normal_mild, moderate, severe = None, None, None, None
            for _row_id, image in loader:
                image = image.to(device)
                outputs = model(image)
                _normal_mild = tuple([el[0].item() for el in outputs])
                _moderate = tuple([el[1].item() for el in outputs])
                _severe = tuple([el[2].item() for el in outputs])
                #print(f"{_row_id}\n{_normal_mild}\n{_moderate}\n{_severe}");1/0
                if row_id is None:
                    row_id, normal_mild, moderate, severe = _row_id, _normal_mild, _moderate, _severe
                else:
                    row_id = row_id + _row_id
                    normal_mild = normal_mild + _normal_mild
                    moderate = moderate + _moderate
                    severe = severe + _severe
            df = pd.DataFrame({
                'row_id':row_id,
                'normal_mild':normal_mild,
                'moderate':moderate,
                'severe':severe,
                })
            return df
        
    def assemble_eval_data(self, foldpath, pred_data):
        def get_prediction(row):
            pred_vec = [row['normal_mild'], row['moderate'], row['severe']]
            predicted_class = pred_vec.index(max(pred_vec))
            return predicted_class
        
        pred_data['prediction'] = pred_data.apply(get_prediction, axis=1)
        fold_data = pd.read_csv(foldpath)
        eval_data = pred_data.merge(fold_data, on='row_id', how='inner')
        return eval_data[['row_id', 'normal_mild', 'moderate', 'severe', 'prediction', 'study_path', 'series_id', 'SeriesDescriptions', 'instance_number', 'condition', 'study_id', 'level', 'x', 'y', 'label']]
        
    def get_cmatrix(self, eval_data):
        return confusion_matrix(eval_data['label'], eval_data['prediction'])

    def get_score(self, eval_data):
        accurate_prediction = (eval_data['label'] == eval_data['prediction'])
        score = 1 - accurate_prediction.mean()
        return score











