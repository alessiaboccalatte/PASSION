import passion
import argparse, pathlib, yaml, pathlib, shapefile
from passion.segmentation import training_slopes

parser = argparse.ArgumentParser()
parser.add_argument('--config', metavar='C', type=str, help='Config file path', default='/scratch/clear/aboccala/PASSION/workflow/config.yml')
args = vars(parser.parse_args())
configfile = args['config']

with open(configfile, "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit

training_config = config.get('SectionSegmentationTraining')
results_path = pathlib.Path(config.get('results_path'))

train_path = pathlib.Path(training_config['train_folder'])
val_path = pathlib.Path(training_config['val_folder'])
model_output_path = pathlib.Path(training_config['output_folder']) / training_config['folder_name']
model_output_path = results_path / model_output_path
model_name = training_config['model_name']

# batch_size = int(training_config['batch_size'])
# n_epochs = int(training_config['n_epochs'])
# learning_rate = float(training_config['learning_rate'])
# num_classes = int(training_config['num_classes'])

batch_size = int(training_config['batch_size'])
n_epochs = int(training_config['n_epochs'])
learning_rate_seg = float(training_config['learning_rate_seg'])
learning_rate_slope = float(training_config['learning_rate_slope'])
num_classes_orient = int(training_config['num_classes_orient'])
num_classes_slope = int(training_config['num_classes_slope'])



# passion.segmentation.training_slopes.train_model(train_path,
#                                           val_path,
#                                           model_output_path,
#                                           model_name,
#                                           num_classes=num_classes,
#                                           batch_size=batch_size,
#                                           learning_rate=learning_rate,
#                                           n_epochs=n_epochs)

passion.segmentation.training_slopes.train_model(train_path,
                                          val_path,
                                          model_output_path,
                                          model_name,
                                          num_classes_orient=num_classes_orient,
                                          num_classes_slope=num_classes_slope,
                                          batch_size=batch_size,
                                          learning_rate_seg=learning_rate_seg,
                                          learning_rate_slope=learning_rate_slope,
                                          n_epochs=n_epochs)

# old
# passion.segmentation.training.train_model(train_path,
#                                           val_path,
#                                           model_output_path,
#                                           model_name,
#                                           num_classes=num_classes,
#                                           batch_size=batch_size,
#                                           learning_rate=learning_rate,
#                                           n_epochs=n_epochs)