using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIManager : MonoBehaviour {

	public Slider particleCountSlider;
	public Slider particleViscositySlider;

	public Slider containerHeightSlider;
	public Slider containerBaseSlider;
	public Slider containerElasticitySlider;

	private Text particleCountSliderText;
	private Text particleViscositySliderText;
	private Text containerHeightSliderText;
	private Text containerBaseSliderText;
	private Text containerElasticitySliderText;

	// Use this for initialization
	void Start () {
		particleCountSliderText = particleCountSlider.GetComponentInChildren<Text> ();
		particleViscositySliderText = particleViscositySlider.GetComponentInChildren<Text> ();
		containerBaseSliderText = containerBaseSlider.GetComponentInChildren<Text> ();
		containerHeightSliderText = containerHeightSlider.GetComponentInChildren<Text> ();
		containerElasticitySliderText = containerElasticitySlider.GetComponentInChildren<Text> ();
	}
	
	// Update is called once per frame
	void Update () {

	}

	public void OnPCSliderValueChange()
	{
		FluidController.Instance.DestroyParticles ();
		FluidController.Instance.particleCount = (uint)particleCountSlider.value;
		particleCountSliderText.text = "Number of particles: " + particleCountSlider.value.ToString();
	}

	public void OnPVSliderValueChange()
	{
		FluidController.Instance.baseObject.viscosity = particleViscositySlider.value;
		particleViscositySliderText.text = "Particle viscosity: " + particleViscositySlider.value.ToString ("0.00") + " mPa * s";
	}

	public void OnCHSliderValueChange()
	{
		FluidController.Instance.container.containerHeight = containerHeightSlider.value;
		containerHeightSliderText.text = "Container Height: " + containerHeightSlider.value.ToString() + " cm";
	}

	public void OnCBSliderValueChange()
	{
		FluidController.Instance.container.containerBase = containerBaseSlider.value;
		containerBaseSliderText.text = "Container Base: " + containerBaseSlider.value.ToString() + " cm";
	}

	public void OnCESliderValueChange()
	{
		FluidController.Instance.container.elasticity = (double)containerElasticitySlider.value;
		containerElasticitySliderText.text = "Container Elasticity: " + containerElasticitySlider.value.ToString () + " GPa";
	}

	public void OnGenerateButtonClick()
	{
		FluidController.Instance.CreateParticles ();
	}
}
