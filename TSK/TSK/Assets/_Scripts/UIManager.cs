using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class UIManager : MonoBehaviour {

	public Slider particleCountSlider;

	private Text particleCountSliderText;

	// Use this for initialization
	void Start () {
		particleCountSliderText = particleCountSlider.GetComponentInChildren<Text> ();
		particleCountSliderText.text = "Number of particles: 50";
	}
	
	// Update is called once per frame
	void Update () {

	}

	public void OnSliderValueChange()
	{
		FluidController.Instance.DestroyParticles ();
		FluidController.Instance.particleCount = (uint)particleCountSlider.value;
		FluidController.Instance.particles = new FluidParticle[(int)particleCountSlider.value];
		particleCountSliderText.text = "Number of particles: " + particleCountSlider.value.ToString();
	}

	public void OnGenerateButtonClick()
	{
		FluidController.Instance.CreateParticles ();
	}
}
