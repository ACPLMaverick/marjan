﻿using UnityEngine;
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
    public Dropper dropper;
	public InteractiveObject baseInteractiveObject;
	public List<InteractiveObject> objects = new List<InteractiveObject>();

	public GameObject initialPosition;
	public uint IDController;
	public bool canDelete;

	public bool startSimulation = false;
	public float particleOffsetX;
	public float particleOffsetY;


    private uint particleWidth;

    #region simRelated

    private bool vfInitialized = false;
    private Color[] vectorField;
    private Color[] vectorFieldNew;
    private Texture2D vectorFieldTexture;

    #endregion

    #region main

    protected FluidController() { }

	public void Start()
	{
		particleCount = 1024;
		particles = new FluidParticle[1024];
		IDController = 0;
		particleOffsetX = 0.25f;
		particleOffsetY = 0.25f;
	}

	public void FixedUpdate()
	{
        if(vfInitialized)
        {
            CalculateVectorField();
            ApplyVectorField();
        }
	}

    #endregion

    #region sim

    public void InitializeVectorField()
    {
        vectorFieldTexture = new Texture2D((int)particleWidth, (int)particleWidth);
        vectorFieldTexture.wrapMode = TextureWrapMode.Clamp;
        vectorField = new Color[particleCount];
        vectorFieldNew = new Color[particleCount];

        for (uint i = 0; i < particleCount; ++i )
        {
            vectorField[i] = new Color(0.0f, 0.0f, 0.0f);
            vectorFieldNew[i] = new Color(0.0f, 0.0f, 0.0f);
        }

        ApplyTextureData(ref vectorFieldTexture, vectorField);
        MaterialPropertyBlock mp = new MaterialPropertyBlock();
        mp.AddTexture(0, vectorFieldTexture);
        container.MySprite.SetPropertyBlock(mp);

        vfInitialized = true;
    }

    private void CalculateVectorField()
    {
        Advect();
        Diffuse();
        ApplyForces();
        ComputePressure();
        SubtractPressureGradient();

        ApplyTextureData(ref vectorFieldTexture, vectorField);

        ApplyVectorField();
    }

    private void Advect()
    {
        for (uint i = 0; i < particleCount; ++i)
        {
            Vector2 cPos = Square1DCoords(i, particleWidth);
            Vector2 backPos = cPos - Time.fixedDeltaTime * new Vector2(vectorField[i].r, vectorField[i].g);

            backPos.x = Mathf.Clamp(backPos.x, 0.0f, (float)particleWidth);
            backPos.y = Mathf.Clamp(backPos.y, 0.0f, (float)particleWidth);

            Vector2 tl, tr, br, bl;
            tl = new Vector2(Mathf.Floor(backPos.x), Mathf.Ceil(backPos.y));
            tr = new Vector2(Mathf.Ceil(backPos.x), Mathf.Ceil(backPos.y));
            br = new Vector2(Mathf.Ceil(backPos.x), Mathf.Floor(backPos.y));
            bl = new Vector2(Mathf.Floor(backPos.x), Mathf.Floor(backPos.y));

            Color newVelocity =
                vectorField[Flatten2DCoords(tl, particleWidth)] +
                vectorField[Flatten2DCoords(tr, particleWidth)] +
                vectorField[Flatten2DCoords(br, particleWidth)] +
                vectorField[Flatten2DCoords(bl, particleWidth)];
            newVelocity /= 4.0f;

            vectorFieldNew[i] = newVelocity;
        }

        SwapColor(ref vectorField, ref vectorFieldNew);
    }

    private void Diffuse()
    {
        
    }

    private void ApplyForces()
    {
        if(dropper.Active)
        {
            Vector2 forceDir = dropper.CurrentForceDirection * dropper.ForceValue * Time.fixedDeltaTime;
            Vector2 forcePos = dropper.CurrentForcePosition;
            Vector2 vel = Vector2.zero;
            for(uint i = 0; i < particleCount; ++i)
            {
                float divisor = (Mathf.Pow(particles[i].transform.position.x - forcePos.x, 2.0f) + 
                        Mathf.Pow(particles[i].transform.position.y - forcePos.y, 2.0f));
                if(divisor == 0.0f)
                    continue;
                vel = forceDir * Mathf.Exp(
                    dropper.Radius /
                    divisor
                    );

                vectorField[i].r = Mathf.Clamp(vectorField[i].r + vel.x * dropper.InsertedDensity, 0.0f, 1.0f);
                vectorField[i].g = Mathf.Clamp(vectorField[i].g + vel.y * dropper.InsertedDensity, 0.0f, 1.0f);
            }
            //Debug.Log(forcePos);
        }
        
    }

    private void ComputePressure()
    {
        
    }

    private void SubtractPressureGradient()
    {
        
    }

    private void ApplyVectorField()
    {

    }

    private void ApplyTextureData(ref Texture2D tex, Color[] field)
    {
        tex.SetPixels(field);
        tex.Apply();
    }

    private void SwapTextureContent(ref Texture2D first, ref Texture2D second)
    {

    }

    private void SwapColorContent(ref Color[] first, ref Color[] second, uint length)
    {
        Color buffer;
        for(uint i = 0; i < length; ++i)
        {
            buffer = second[i];
            second[i] = first[i];
            first[i] = buffer;
        }
    }

    private void SwapColor(ref Color[] first, ref Color[] second)
    {
        Color[] temp = first;
        first = second;
        second = temp;
    }

    private uint Flatten2DCoords(uint i, uint j, uint width)
    {
        return i * width + j;
    }

    private uint Flatten2DCoords(Vector2 coord, uint width)
    {
        return (uint)coord.x * width + (uint)coord.y;
    }

    private Vector2 Square1DCoords(uint coord, uint width)
    {
        Vector2 ret = Vector2.zero;

        ret.x = coord / width;
        ret.y = coord % width;

        return ret;
    }

    #endregion

    #region creation

    public void CreateParticles()
	{
		DestroyParticles ();
		particles = new FluidParticle[particleCount];

		float x = initialPosition.transform.position.x;
		float y = initialPosition.transform.position.y;

		particleWidth = (uint)Mathf.Sqrt (particleCount);

		for (uint i = 0; i<particleWidth; ++i) {
			for (uint j = 0; j < particleWidth; ++j) {
				particles[j + particleWidth * i] = (FluidParticle)Instantiate (baseObject, new Vector2(x, y), Quaternion.identity);
				particles[j + particleWidth * i].viscosity = baseObject.viscosity;
				particles[j + particleWidth * i].position = particles[j + particleWidth * i].transform.position;
				CalculatePosition(ref x, ref y, particleCount, false);
			}
			CalculatePosition(ref x, ref y, particleCount, true);
		}

        startSimulation = true;
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

    #endregion
}
