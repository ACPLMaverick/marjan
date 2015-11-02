using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections;
using System.Collections.Generic;

public class FluidController : Singleton<FluidController> {

	public uint particleCount;
	public double particleMass;
	public double particleVelocity;

	public FluidParticle[] particles;
	public FluidContainer container;
	public FluidParticle baseObject;
	public InteractiveObject baseInteractiveObject;
	public List<InteractiveObject> objects = new List<InteractiveObject>();

	public GameObject initialPosition;
	public uint IDController;
	public bool canDelete;

	public bool startSimulation = false;
	public float particleOffsetX;
	public float particleOffsetY;

	protected FluidController() { }

	public void Start()
	{
		particleCount = 1024;
		particles = new FluidParticle[1024];
		IDController = 0;
		particleOffsetX = 0.25f;
		particleOffsetY = 0.25f;
	}

	public void Update()
	{

	}

	public void CreateParticles()
	{
		DestroyParticles ();
		particles = new FluidParticle[particleCount];

		float x = initialPosition.transform.position.x;
		float y = initialPosition.transform.position.y;

		int width = (int)Mathf.Sqrt (particleCount);

		for (int i = 0; i<width; ++i) {
			for (int j = 0; j < width; ++j) {
				particles[j + width * i] = (FluidParticle)Instantiate (baseObject, new Vector2(x, y), Quaternion.identity);
				particles[j + width * i].viscosity = baseObject.viscosity;
				particles[j + width * i].position = particles[j + width * i].transform.position;
				CalculatePosition(ref x, ref y, particleCount, false);
			}
			CalculatePosition(ref x, ref y, particleCount, true);
		}
	}

	public void CalculatePosition(ref float inputX, ref float inputY, uint count, bool moveUp)
	{
		switch (count) {
		case 1024:
			particleOffsetX = 0.285f;
			particleOffsetY = 0.285f;
			break;
		case 4096:
			particleOffsetX = 0.143f;
			particleOffsetY = 0.143f;
			break;
		case 16384:
			particleOffsetX = 0.072f;
			particleOffsetY = 0.072f;
			break;
		}

		inputX += particleOffsetX;
		if (moveUp) {
			inputX = initialPosition.transform.position.x;
			inputY += particleOffsetY;
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

	public void DestroyInteractiveObject(InteractiveObject io)
	{
		Debug.Log ("Destroy");

		objects.Remove (io);
		Destroy (io.gameObject);

		canDelete = false;
	}
}
