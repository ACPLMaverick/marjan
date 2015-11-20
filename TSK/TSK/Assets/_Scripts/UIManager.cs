using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIManager : MonoBehaviour {

	public Slider particleCountSlider;
	public Slider particleViscositySlider;
	public Slider dissipationSlider;
	public Slider jacobiIterationsSlider;

	public Button dropperSettingsButton;
	public Button generateButton;
	public Button startSimulationButton;

	public Animator myAnimator;

	public GameObject dropperSettingsPanel;
	public Slider dropperRadiusSlider;
	public Slider dropperForceSlider;
	public Slider dropperInsertedDensitySlider;

	public Dropper dropper;

	private Text particleCountSliderText;
	private Text particleViscositySliderText;
	private Text dissipationSliderText;
	private Text jacobiIterationsSliderText;

	private Text dropperRadiusSliderText;
	private Text dropperForceSliderText;
	private Text dropperInsertedDensitySliderText;

	private Text startSimulationButtonText;

	private float positionX, positionY;
	private bool settingPosition = false;
	private bool deletingObject = false;

	// Use this for initialization
	void Start () {
		particleCountSliderText = particleCountSlider.GetComponentInChildren<Text> ();
		particleViscositySliderText = particleViscositySlider.GetComponentInChildren<Text> ();
		dissipationSliderText = dissipationSlider.GetComponentInChildren<Text> ();
		jacobiIterationsSliderText = jacobiIterationsSlider.GetComponentInChildren<Text> ();

		dropperRadiusSliderText = dropperRadiusSlider.GetComponentInChildren<Text> ();
		dropperForceSliderText = dropperForceSlider.GetComponentInChildren<Text> ();
		dropperInsertedDensitySliderText = dropperInsertedDensitySlider.GetComponentInChildren<Text> ();

		startSimulationButton.interactable = false;
		startSimulationButtonText = startSimulationButton.GetComponentInChildren<Text> ();

		positionX = 0;
		positionY = 0;
	}
	
	// Update is called once per frame
	void Update () {
		UpdateStrings ();
	}
	
	void UpdateStrings()
	{
		particleCountSliderText.text = "Particle count: " + SetParticleCount ().ToString();
		particleViscositySliderText.text = "Particle viscosity: " + particleViscositySlider.value.ToString ("0.00") + " mPa * s";
		dissipationSliderText.text = "Dissipation: " + dissipationSlider.value.ToString ("0.000");
		jacobiIterationsSliderText.text = "Jacobi Iterations: " + jacobiIterationsSlider.value.ToString ();

		if (dropperSettingsPanel.activeInHierarchy) {
			dropperRadiusSliderText.text = "Dropper radius: " + dropperRadiusSlider.value.ToString ("0.0");
			dropperForceSliderText.text = "Dropper force: " + dropperForceSlider.value.ToString ("0.00");
			dropperInsertedDensitySliderText.text = "Dropper density: " + dropperInsertedDensitySlider.value.ToString ("0.0000");
		}
	}

	public void OnPCSliderValueChange()
	{
		FluidControllerGPU.Instance.DestroyParticles ();
		FluidControllerGPU.Instance.particleCount = (uint)SetParticleCount ();
		switch (SetParticleCount ()) {
		case 1024:
			FluidControllerGPU.Instance.baseObject.transform.localScale = new Vector3(1, 1, 1);
			FluidControllerGPU.Instance.initialPosition.transform.localPosition = new Vector2(-1.13f, -1.13f);
			break;
		case 4096:
			FluidControllerGPU.Instance.baseObject.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
			FluidControllerGPU.Instance.initialPosition.transform.localPosition = new Vector2(-1.15f, -1.15f);
			break;
		case 16384:
			FluidControllerGPU.Instance.baseObject.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);
			FluidControllerGPU.Instance.initialPosition.transform.localPosition = new Vector2(-1.17f, -1.17f);
			break;
		}
	}

	public void OnDSliderValueChange()
	{
		FluidControllerGPU.Instance.DISSIPATION = (float)dissipationSlider.value;
	}

	public void OnJISliderValueChange()
	{
		FluidControllerGPU.Instance.JACOBI_ITERATIONS = (uint)jacobiIterationsSlider.value;
	}

	int SetParticleCount()
	{
		int i = 0;
		switch ((int)particleCountSlider.value) {
		case 0:
			i = 1024;
			break;
		case 1:
			i = 4096;
			break;
		case 2:
			i = 16384;
			break;
		}
		return i;
	}

	public void OnPVSliderValueChange()
	{
		FluidControllerGPU.Instance.baseObject.viscosity = particleViscositySlider.value;
	}

	public void OnDropperSettingsButtonClick()
	{
		dropperSettingsPanel.SetActive (true);
	}

	public void SaveDropperSettings()
	{
		dropper.Radius = dropperRadiusSlider.value;
		dropper.ForceValue = dropperForceSlider.value;
		dropper.InsertedDensity = dropperInsertedDensitySlider.value;
		dropperSettingsPanel.SetActive (false);
	}

	public void OnGenerateButtonClick()
	{
        FluidControllerGPU.Instance.CreateParticles();
        FluidControllerGPU.Instance.InitializeVectorField();
		startSimulationButton.interactable = true;
	}

	public void SimulationButtonOnClick()
	{
		if (!FluidControllerGPU.Instance.startSimulation) {
			ChangeInteractionUI(false);
			startSimulationButtonText.text = "Stop";
			FluidControllerGPU.Instance.startSimulation = true;
		}
		else {
			startSimulationButton.interactable = false;
			startSimulationButtonText.text = "Simulate";
			ChangeInteractionUI(true);
			FluidControllerGPU.Instance.startSimulation = false;
		}
	}

	void ChangeInteractionUI(bool value)
	{
		particleCountSlider.interactable = value;
		particleViscositySlider.interactable = value;
		dissipationSlider.interactable = value;
		jacobiIterationsSlider.interactable = value;
		dropperSettingsButton.interactable = value;
		generateButton.interactable = value;
	}

	public void ShowUI()
	{
		myAnimator.SetBool ("Show", true);
	}

	public void HideUI()
	{
		myAnimator.SetBool ("Show", false);
	}
}