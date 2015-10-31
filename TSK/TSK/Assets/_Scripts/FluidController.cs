using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class FluidController : Singleton<FluidController> {

	public uint particleCount;
	public double particleMass;
	public double particleVelocity;

	public FluidParticle[] particles;
	public FluidContainer container;
	public FluidParticle baseObject;
	public List<InteractiveObject> objects = new List<InteractiveObject>();

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
		float j = 0;
		for (int i = 0; i < particleCount; i++) {
			if (i % 10 == 0) {
				j += 0.25f;
			}
			particles [i] = (FluidParticle)Instantiate (baseObject, new Vector3 (j - 4.5f, (i % 10) - (1.5f + 0.75f * (i % 10))), Quaternion.identity);
			particles [i].viscosity = baseObject.viscosity;
			particles [i].position = particles [i].transform.position;
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
