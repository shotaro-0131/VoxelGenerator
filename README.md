蛋白質の立体構造を考慮した化合物生成手法に関する研究
====

標的タンパク質のPDBファイルから新規化合物のPDBファイルを生成する．

# Requirements
Python Package 

Common:
- python>=3.6.9
- numpy>=1.19.2
- hydra>=2.5
- pymol>=2.3.5

Machine Learning:
- pytorch>=1.10.1
- pytorch-lightning>=1.5.7
- mlflow>=1.17.0
- optuna>=2.8.0

Generator(MCTS):
- openbabel>=3.1.1
- vina>=1.2.3
- rdkit>=2020.09.1.0
- mcts https://pypi.org/project/mcts/
  
## Usage
[Hydra](https://hydra.cc/docs/intro//)でパラメータを管理しています．<br>
- [params.yaml](params.yaml): Machine Learning関連のパラメータ管理
- [generator/search_params.yaml](generator/search_params.yml): 生成手法関連のパラメータ管理

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

Predition:<br>
訓練したモデルで予測
```bash
python predict.py # dataset.test.index_file=v2020_test_index.csv dataset.test.data_dir=test_data
```
- v2020_test_index.csv: DUD-Eのデータのタンパク質名
- test_data: 詳細は[Data PreparetionのテストデータDirectory](#テストデータDirectory)を参照．


Molecular Generation:<br>
訓練したモデルの予測値から分子を生成する<br>
Outputs
- target.csv 生成化合物のhash値，vinaスコア，QEDスコア，SAスコア，予測値の和
- tmp/{生成化合物のhash値}.pdb 生成化合物のPDBファイル
```bash
python generator/search.py \
    hydra.run.dir={生成先ディレクトリ名} \
    target={標的分子のファイル名} #必要(詳細についてはData Prepatetionを参照)
```

## Data Preparetion

### 訓練データDirectory
訓練データのDirectoryは[params.yaml](params.yaml)のdataset.data_dirで指定する．

以下のようにデータを配置する．
- ../v2020_PL_all : [PDBBind](http://www.pdbbind.org.cn/)のデータの入ったディレクトリ
  - /{pdbid}/{pdbid}_pocket.pdb : タンパク質のポケットのPDBファイル(Inputに使用)
  - /{pdbid}/{pdbid}_ligand.sdf : リガンドのSDFファイル(Outputに使用)
- ../v2020-points : PDBBindの原子情報をNumpyに加工したデータ．Tsubame上では，Pymolのアカウント数制限があるため，また計算速度を上げるため．
  - /v2020-points-{pdbid}.npy
  
### テストデータDirectory
テストデータのDirectoryは[generator/search_params.yaml](generator/search_params.yml)で指定する．

以下のようにデータを配置する．
- /test_data/{target名} : [DUD-E Diverseサブセットのデータ](http://dude.docking.org/subsets/diverse)
  - /receptor.pdb : タンパク質のPDBファイル(Inputに使用)
  - /receptor.pdbqt : タンパク質のPDBQTファイル(Dockingに使用．詳細は[PDBQTファイルの作り方](#PDBQTファイルの作り方)を参照)
  - /pred_voxel.npy : 機械学習で予測したボクセルデータ([UsageのPrediction](#Usage)で生成)

### PDBQTファイルの作り方
Auto Dock Vinaでドッキングスコアを計算する際には，化合物をPDBQTファイルに変換しなければならない．

Openbabelを使用する．
- リガンドのPDBQTファイルの作成
```bash
obabel {ligand file name} -O {出力先ファイル名}.pdbqt -xh --partialcharge gasteiger
```

- タンパク質のPDBQTファイルの作成
```bash
obabel {protein file name} -O tmp.pdbqt -xh --partialcharge gasteiger
grep ATOM tmp.pdbqt > {出力先ファイル名}.pdbqt
rm tmp.pdbqt
```

実際にドッキングスコアを計算できるか確かめる．
```bash
python score.py {ligand file name}.pdbqt {protein file name}.pdbqt
```

以下のような出力があればOK．
```bash
Computing Vina grid ... done.
Score before minimization: XXXX (kcal/mol)
```

## Results
### 学習済みモデル

### 生成済み化合物
生成した化合物を置くDirectoryは[generator/search_params.yaml](generator/search_params.yml)の
hydra.run.dirで指定する．

指定したDirectoryの下に以下のファイルが出力される
- /{target}
  - /target.csv : 生成化合物のhash値，vinaスコア，QEDスコア，SAスコア，予測値の和
  - /mols/{生成化合物のhash値}.pdb : 生成化合物のPDBファイル