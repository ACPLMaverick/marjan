using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIManager : MonoBehaviour {

	public InputField inputField;
	public Slider particleViscositySlider;

	public Slider containerHeightSlider;
	public Slider containerBaseSlider;
	public Slider containerElasticitySlider;

	private Text particleViscositySliderText;
	private Text containerHeightSliderText;
	private Text containerBaseSliderText;
	private Text containerElasticitySliderText;

	private int particleCountMinBound, particleCountMaxBound;

	// Use this for initialization
	void Start () {
		particleViscositySliderText = particleViscositySlider.GetComponentInChildren<Text> ();
		containerBaseSliderText = containerBaseSlider.GetComponentInChildren<Text> ();
		containerHeightSliderText = containerHeightSlider.GetComponentInChildren<Text> ();
		containerElasticitySliderText = containerElasticitySlider.GetComponentInChildren<Text> ();

		particleCountMinBound = 10;
		particleCountMaxBound = 10;
	}
	
	// Update is called once per frame
	void Update () {
		UpdateStrings ();
	}

	public void OnPVSliderValueChange()
	{
		FluidController.Instance.baseObject.viscosity = particleViscositySlider.value;
	}

	public void OnCHSliderValueChange()
	{
		FluidController.Instance.container.containerHeight = containerHeightSlider.value;
	}

	public void OnCBSliderValueChange()
	{
		FluidController.Instance.container.containerBase = containerBaseSlider.value;
	}

	public void OnCESliderValueChange()
	{
		FluidController.Instance.container.elasticity = (double)containerElasticitySlider.value;
	}

	public void OnGenerateButtonClick()
	{
		FluidController.Instance.CreateParticles ();
	}

	void UpdateStrings()
	{
		particleViscositySliderText.text = "Particle viscosity: " + particleViscositySlider.value.ToString ("0.00") + " mPa * s";
		containerHeightSliderText.text = "Container Height: " + containerHeightSlider.value.ToString() + " cm";
		containerBaseSliderText.text = "Container Base: " + containerBaseSlider.value.ToString() + " cm";
		containerElasticitySliderText.text = "Container Elasticity: " + containerElasticitySlider.value.ToString () + " GPa";
	}

	public void OnInputFieldValueChange()
	{
		particleCountMinBound = 10;
		particleCountMaxBound = 10;
		containerBaseSlider.minValue = 25;
		containerHeightSlider.minValue = 25;
	}


	//NIE DOTYKAĆ NAWET JEŚLI WYGLĄDA PASKUDNIE (chyba że spowoduje spadki plynności w co wątpię)
	public void OnInputFieldEndEdit()
	{
		FluidController.Instance.DestroyParticles ();
		FluidController.Instance.particleCount = (uint)int.Parse(inputField.text);

		if (int.Parse (inputField.text) > 1800)
			inputField.text = "1800";
		if(int.Parse(inputField.text) < 50)
		   	inputField.text = "50";
		
		int i = Mathf.CeilToInt (Mathf.Sqrt (int.Parse(inputField.text)));
		
		if (int.Parse(inputField.text) > 100 /*&& particleCountSlider.value <= 1100*/) {
			if (i > particleCountMaxBound) {
				particleCountMaxBound = i;
				particleCountMinBound = i - 1;
				if(int.Parse(inputField.text) <= 1100)
				{
					containerHeightSlider.minValue += (i - 10) * 3;
					containerBaseSlider.minValue += (i - 10) * 3;
				}
				else
				{
					containerHeightSlider.minValue = 98;
					containerBaseSlider.minValue += (i - 10) * 6;
				}
				if(containerBaseSlider.minValue > 130)
					containerBaseSlider.minValue = 130;
			} else if (i <= particleCountMinBound) {
				particleCountMaxBound = i;
				particleCountMinBound = i - 1;
				if(int.Parse(inputField.text) <= 1100)
				{
					containerHeightSlider.minValue -= (i - 10) * 3;
					containerBaseSlider.minValue -= (i - 10) * 3;
				}
				else
				{
					containerHeightSlider.minValue = 98;
					containerBaseSlider.minValue -= (i - 10) * 6;
				}
			}
		} else if (int.Parse(inputField.text) <= 100) {
			particleCountMinBound = 10;
			particleCountMaxBound = 10;
			containerHeightSlider.minValue = 25;
			containerBaseSlider.minValue = 25;
		}
	}
}
