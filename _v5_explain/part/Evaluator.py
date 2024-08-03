
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
            id, positive, negative = None, None, None
            for _id, image in loader:
                image = image.to(device)
                outputs = model(image)
                _positive = tuple([el[0].item() for el in outputs])
                _negative = tuple([el[1].item() for el in outputs])
                if id is None:
                    id, positive, negative = _id, _positive, _negative
                else:
                    id = id + _id
                    positive = positive + _positive
                    negative = negative + _negative
            df = pd.DataFrame({
                'address':id,
                'positive':positive,
                'negative':negative,
                })
            return df
        
    def assemble_eval_data(self, foldpath, pred_data):
        def get_prediction(row):
            pred_vec = [row['positive'], row['negative']]
            predicted_class = pred_vec.index(max(pred_vec))
            return predicted_class
        
        pred_data['prediction'] = pred_data.apply(get_prediction, axis=1)
        fold_data = pd.read_csv(foldpath)
        eval_data = pred_data.merge(fold_data, on='address', how='inner')
        return eval_data[['ID', 'positive', 'negative', 'prediction', 'label', 'extent', 'growth_stage', 'season', 'damage', 'address']]
        
    def get_cmatrix(self, eval_data):
        return confusion_matrix(eval_data['label'], eval_data['prediction'])

    def get_score(self, eval_data):
        accurate_prediction = (eval_data['label'] == eval_data['prediction'])
        score = 1 - accurate_prediction.mean()
        return score











