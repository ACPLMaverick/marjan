using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class FluidController : Singleton<FluidController> {

	public uint particleCount; //na szerokość zmieści się 51 rzędów particli, na wysokość zmieści się 38 rzędów - wynik: 1938 particli max
	public double particleMass;
	public double particleVelocity;

	public FluidParticle[] particles;
	public FluidContainer container;
	public FluidParticle baseObject;
	public List<InteractiveObject> objects = new List<InteractiveObject>();

	public GameObject initialPosition;

	protected FluidController() { }

	public void Start()
	{
		particleCount = 50;
		particles = new FluidParticle[50];
	}

	public void Update()
	{

	}

	public void CreateParticles()
	{
		DestroyParticles ();
		particles = new FluidParticle[particleCount];
		//float x = -container.transform.localScale.x;
		//float y = -container.transform.localScale.y;

		float x = initialPosition.transform.position.x;
		float y = initialPosition.transform.position.y;

		for (int i = 0; i < particleCount; i++) {
			particles [i] = (FluidParticle)Instantiate (baseObject, new Vector2(x, y), Quaternion.identity);
			particles [i].viscosity = baseObject.viscosity;
			particles [i].position = particles [i].transform.position;
			CalculatePosition(ref x, ref y);
		}
	}

	public void CalculatePosition(ref float inputX, ref float inputY)
	{
		float i = inputX + 0.25f;

		if (i >= container.MySprite.bounds.max.x - 0.25f) {
			inputX = initialPosition.transform.position.x;
			inputY += 0.25f;
		} else {
			inputX += 0.25f;
		}
	}

	public void DestroyParticles()
	{
		if (particles [0] != null) {
			for (int i = 0; i < particles.Length; i++) {
				Destroy (particles [i].gameObject);
			}
		}
	}
}
