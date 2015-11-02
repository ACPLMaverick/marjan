using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIManager : MonoBehaviour {

	public Slider particleCountSlider;
	public Slider particleViscositySlider;

	//public Slider containerHeightSlider;
	//public Slider containerBaseSlider;
	public Slider containerElasticitySlider;

	public GameObject interactiveObjectPanel;
	public Slider interactiveObjectTypeSlider;
	public Slider interactiveObjectVelocitySlider;
	public Slider interactiveObjectMassSlider;

	public Button addObjectButton;
	public Button deleteObjectButton;
	public Button generateButton;
	public Button startSimulationButton;

	public Text currentPositionText;
	public Animator myAnimator;

	private Text particleCountSliderText;
	private Text particleViscositySliderText;
	//private Text containerHeightSliderText;
	//private Text containerBaseSliderText;
	private Text containerElasticitySliderText;

	private Text interactiveObjectTypeSliderText;
	private Text interactiveObjectVelocitySliderText;
	private Text interactiveObjectMassSliderText;

	private Text startSimulationButtonText;

	//private int particleCountMinBound, particleCountMaxBound;
	private float positionX, positionY;
	private bool settingPosition = false;
	private bool deletingObject = false;

	// Use this for initialization
	void Start () {
		particleCountSliderText = particleCountSlider.GetComponentInChildren<Text> ();
		particleViscositySliderText = particleViscositySlider.GetComponentInChildren<Text> ();
		//containerBaseSliderText = containerBaseSlider.GetComponentInChildren<Text> ();
		//containerHeightSliderText = containerHeightSlider.GetComponentInChildren<Text> ();
		containerElasticitySliderText = containerElasticitySlider.GetComponentInChildren<Text> ();

		startSimulationButton.interactable = false;
		startSimulationButtonText = startSimulationButton.GetComponentInChildren<Text> ();

		//particleCountMinBound = 10;
		//particleCountMaxBound = 10;

		positionX = 0;
		positionY = 0;
	}
	
	// Update is called once per frame
	void Update () {
		UpdateStrings ();

		if (settingPosition) {
			if (Input.GetMouseButtonDown (0)) {
				positionX = Camera.main.ScreenToWorldPoint(Input.mousePosition).x;
				positionY = Camera.main.ScreenToWorldPoint(Input.mousePosition).y;
				interactiveObjectPanel.SetActive (true);
				settingPosition = false;
			}
		}

		if (deletingObject) {
			//FluidController.Instance.DestroyInteractiveObject();
			if(!FluidController.Instance.canDelete)
				deletingObject = false;
		}
	}
	
	void UpdateStrings()
	{
		particleCountSliderText.text = "Particle count: " + SetParticleCount ().ToString();
		particleViscositySliderText.text = "Particle viscosity: " + particleViscositySlider.value.ToString ("0.00") + " mPa * s";
		//containerHeightSliderText.text = "Container Height: " + containerHeightSlider.value.ToString() + " cm";
		//containerBaseSliderText.text = "Container Base: " + containerBaseSlider.value.ToString() + " cm";
		containerElasticitySliderText.text = "Container Elasticity: " + containerElasticitySlider.value.ToString () + " GPa";

		if(interactiveObjectVelocitySliderText != null)
			interactiveObjectVelocitySliderText.text = "Object Velocity: " + interactiveObjectVelocitySlider.value.ToString ("0.00") + " m/s";
		if(interactiveObjectMassSliderText != null)
			interactiveObjectMassSliderText.text = "Object Mass: " + interactiveObjectMassSlider.value.ToString () + " g";
		if (interactiveObjectTypeSliderText != null) {
			switch ((int)interactiveObjectTypeSlider.value) {
			case 0:
				interactiveObjectTypeSliderText.text = "Object Type: Square";
				break;
			case 1:
				interactiveObjectTypeSliderText.text = "Object Type: Circle";
				break;
			}
		}
		if (currentPositionText != null)
			currentPositionText.text = positionX.ToString ("0.0") + ", " + positionY.ToString ("0.0");
	}

	public void OnPCSliderValueChange()
	{
		FluidController.Instance.DestroyParticles ();
		FluidController.Instance.particleCount = (uint)SetParticleCount ();
		switch (SetParticleCount ()) {
		case 1024:
			FluidController.Instance.baseObject.transform.localScale = new Vector3(1, 1, 1);
			FluidController.Instance.initialPosition.transform.localPosition = new Vector2(-1.13f, -1.13f);
			break;
		case 4096:
			FluidController.Instance.baseObject.transform.localScale = new Vector3(0.5f, 0.5f, 0.5f);
			FluidController.Instance.initialPosition.transform.localPosition = new Vector2(-1.15f, -1.15f);
			break;
		case 16384:
			FluidController.Instance.baseObject.transform.localScale = new Vector3(0.3f, 0.3f, 0.3f);
			FluidController.Instance.initialPosition.transform.localPosition = new Vector2(-1.17f, -1.17f);
			break;
		}
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
		FluidController.Instance.baseObject.viscosity = particleViscositySlider.value;
	}

//	public void OnCHSliderValueChange()
//	{
//		FluidController.Instance.container.containerHeight = containerHeightSlider.value;
//	}
//
//	public void OnCBSliderValueChange()
//	{
//		FluidController.Instance.container.containerBase = containerBaseSlider.value;
//	}

	public void OnCESliderValueChange()
	{
		FluidController.Instance.container.elasticity = (double)containerElasticitySlider.value;
	}

	public void OnGenerateButtonClick()
	{
		FluidController.Instance.CreateParticles ();
		startSimulationButton.interactable = true;
		//if (inputField.placeholder != null)
		//	inputField.text = "50";
	}

	public void OpenInteractiveObjectSettings()
	{
		interactiveObjectPanel.SetActive (true);

		interactiveObjectTypeSliderText = interactiveObjectTypeSlider.GetComponentInChildren<Text> ();
		interactiveObjectVelocitySliderText = interactiveObjectVelocitySlider.GetComponentInChildren<Text> ();
		interactiveObjectMassSliderText = interactiveObjectMassSlider.GetComponentInChildren<Text> ();
	}

	public void CloseInteractiveObjectSettings()
	{
		interactiveObjectPanel.SetActive (false);
	}

	public void SetPosition()
	{
		interactiveObjectPanel.SetActive (false);
		settingPosition = true;
	}

	public void AddInteractiveObject()
	{
		FluidController.Instance.baseInteractiveObject.SetSprite ((int)interactiveObjectTypeSlider.value);
		FluidController.Instance.baseInteractiveObject.velocity = (double)interactiveObjectVelocitySlider.value;
		FluidController.Instance.baseInteractiveObject.mass = (double)interactiveObjectMassSlider.value;
		FluidController.Instance.baseInteractiveObject.ID = FluidController.Instance.IDController;

		InteractiveObject obj = (InteractiveObject)Instantiate (FluidController.Instance.baseInteractiveObject,
		                                                        new Vector2(positionX, positionY),
		                                                        Quaternion.identity);

		FluidController.Instance.objects.Add (obj);
		FluidController.Instance.IDController++;

		interactiveObjectPanel.SetActive (false);
	}

	public void DeleteInteractiveObject()
	{
		interactiveObjectPanel.SetActive (false);
		deletingObject = true;
		FluidController.Instance.canDelete = true;
	}

	public void SimulationButtonOnClick()
	{
		if (!FluidController.Instance.startSimulation) {
			ChangeInteractionUI(false);
			interactiveObjectPanel.SetActive(false);
			startSimulationButtonText.text = "Stop";
			FluidController.Instance.startSimulation = true;
		}
		else {
			startSimulationButton.interactable = false;
			startSimulationButtonText.text = "Simulate";
			ChangeInteractionUI(true);
			FluidController.Instance.startSimulation = false;
		}
	}

	void ChangeInteractionUI(bool value)
	{
		//inputField.interactable = value;
		particleCountSlider.interactable = value;
		particleViscositySlider.interactable = value;
		//containerHeightSlider.interactable = value;
		//containerBaseSlider.interactable = value;
		containerElasticitySlider.interactable = value;
		addObjectButton.interactable = value;
		deleteObjectButton.interactable = value;
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