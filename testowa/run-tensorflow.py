# Control script
from azureml.core import Experiment, Environment, ScriptRunConfig
from azureml.core import Workspace

if __name__=="__main__":
    ws = Workspace.from_config()

    cpu_cluster_name = "cpu-cluster"
    experiment = Experiment(workspace=ws, name='day2-experiment-train')

    config=ScriptRunConfig(source_directory='.',  script='train.py', compute_target=cpu_cluster_name)
    
    # set up python env
    env = Environment.from_conda_specification(name='keras-env', file_path='.azureml/keras-env.yml')

    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)