from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import DefaultAzureCredential

# ==========================================
# 1. Configuration & Job Mapping
# ==========================================
TENANT_ID       = "ebbb4fc3-587b-477c-a1e8-ee47d8c02546"
SUBSCRIPTION_ID = "ab211f7b-463f-4833-9605-d260e596a35a"
RESOURCE_GROUP  = "73da10b4-5dff-54e2-db0d-3a1fab882485"
WORKSPACE_NAME  = "73da10b45dff54e2db0d3a1fab882485"

# The 5 successful training jobs to validate
cv_job_names = {
    "w0": "gifted_leek_gsn13rz9y0",
    "w1": "magenta_stomach_rp773hscbs",
    "w2": "polite_shampoo_drv30v29hs",
    "w3": "olden_cat_bskn3cdb3g",
    "w4": "ashy_rail_snvq74fr39"
}

# ==========================================
# 2. Authentication & Client Setup
# ==========================================
print("Opening browser for authentication...")
# credential = InteractiveBrowserCredential(tenant_id=TENANT_ID)
credential = DefaultAzureCredential()

ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME,
)

# ==========================================
# 3. Dynamic Input Construction
# ==========================================
# Start with the static data assets
job_inputs = {
    "input_data": Input(type="uri_folder", path="azureml:LEMS-CT-NIfTI:1"),
    "split_csv": Input(
        type="uri_file", 
        path="./data/cv_splits.csv"
    ),
}

# Programmatically add the weights folders for all 5 folds
for key, job_id in cv_job_names.items():
    job_inputs[key] = Input(
        type="uri_folder",
        path=f"azureml:azureml_{job_id}_output_data_output_model:1",
        mode="ro_mount"
    )

# ==========================================
# 4. Define the Command Job
# ==========================================
validation_job = command(
    display_name="Myocardium Full CV Validation",
    description="Validates all 5 folds sequentially.",
    experiment_name="myocardium_val_full_cv",
    compute="vmprdwe1-gpu-vzhst6",
    environment="azureml:media-env:7",
    
    # Path to root of project
    code=".", 
    environment_variables={"PYTHONPATH": "."},
    inputs=job_inputs,
    outputs={
        "validation_results": Output(type="uri_folder")
    },
    
    # The command string uses placeholders that match our inputs_dict keys
    command=(
        "python scripts/validation_all_folds.py "
        "--input_data ${{inputs.input_data}} "
        "--split_csv ${{inputs.split_csv}} "
        "--config config/train_config.yaml "
        "--output_dir ${{outputs.validation_results}} "
        "--w0 ${{inputs.w0}} --w1 ${{inputs.w1}} --w2 ${{inputs.w2}} --w3 ${{inputs.w3}} --w4 ${{inputs.w4}}"
    ),
    
    resources={"instance_count": 1}
)

# ==========================================
# 5. Submit and Track
# ==========================================
print("\nSubmitting job to Azure ML...")
returned_job = ml_client.jobs.create_or_update(validation_job)

print("--------------------------------------------------")
print(f"✅ Job submitted! ID: {returned_job.name}")
print(f"Check status here: {returned_job.studio_url}")
print("--------------------------------------------------")