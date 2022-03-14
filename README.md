蛋白質の立体構造を考慮した化合物生成手法に関する研究
====

標的タンパク質のPDBファイルから新規化合物のPDBファイルを生成する．

# Requirements
Python Package 

Common:
- python 3.6.9
- numpy 1.19.2
- hydra 2.5
- pymol 2.3.5

Machine Learning:
- pytorch 1.10.1
- pytorch-lightning 1.5.7
- mlflow 1.17.0
- optuna 2.8.0

Generator(MCTS):
- openbabel 3.1.1
- vina 1.2.3
- rdkit 2020.09.1.0
- mcts
  
## Usage
Hyperparameters Optimization:<br>
実行結果とスコアがDBに保存される
```bash
python model_fine.py training.batch_size={バッチサイズ} training.gpu_num={GPUの数}
```
Training:<br>
最適化したハイパーパラメータで訓練
```bash
python model_train.py
```
Molecular Generation:<br>
訓練したモデルの予測値から分子を生成する<br>
Outputs
- target.csv 生成化合物のhash値，vinaスコア，QEDスコア，SAスコア，予測値の和
- tmp/{生成化合物のhash値}.pdb 生成化合物のPDBファイル
```bash
python generator/search.py \
    hydra.run.dir={生成先ディレクトリ名} \
    target={標的分子のファイル名} #defalt ace
```

## Data Preparetion
Machine Learning:

PDBBind
proteins pdbファイル
ligands pdbファイル