from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location="score.py",
    name="ride_duration_prediction",
    parameters={'taxi_type': 'green', 'run_id': 'ce361da39ca847d7b993f11d7b1dbec6'},
    flow_storage='20a3b87d-2e42-447b-8f39-837bc9b3de36',
    flow_runner=SubprocessFlowRunner(),
    schedule=CronSchedule(cron="0 3 2 * *"),
    tags=["batch"]
)
