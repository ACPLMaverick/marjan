using UnityEngine;
using UnityEngine.EventSystems;
using System.Collections;
using System.Collections.Generic;



public class FluidControllerGPU : Singleton<FluidControllerGPU>
{

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

    const uint JACOBI_ITERATIONS = 20;
    const uint RT_COUNT = 6;
    const int THREAD_COUNT = 32;

    public ComputeShader cShader;

    private enum KernelIDs
    {
        ADVECT,
        DIFFUSE,
        APPLY_FORCE,
        PROJECT,
        SWAP_TO_NEW,
        SWAP_TO_OLD,
        INITIALIZE
    }

    private int[] kernelIDs = new int[7];

    private bool vfInitialized = false;
    private ComputeBuffer velocityField;
    private ComputeBuffer velocityFieldNew;
    private ComputeBuffer pressureField;
    private ComputeBuffer pressureFieldNew;
    private ComputeBuffer jacobiHelper;
    private ComputeBuffer particleData;

    private ComputeBuffer[] cbArray = new ComputeBuffer[RT_COUNT];

    private Texture2D finalTexture;
    private Vector2[] velocityFieldBuffer;
    private Vector4[] particleDataBuffer;

    #endregion

    #region main

    protected FluidControllerGPU() { }

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
        if (vfInitialized)
        {
            CalculateVectorField();
            ApplyVectorField();
        }
    }
    /*
    public void OnDisable()
    {
        velocityField.Dispose();
        velocityFieldNew.Dispose();
        pressureField.Dispose();
        pressureFieldNew.Dispose();
        jacobiHelper.Dispose();
        particleData.Dispose();
    }
    */
    #endregion

    #region sim

    public void InitializeVectorField()
    {
        velocityField = new ComputeBuffer((int)particleCount, 2 * sizeof(float));
        velocityFieldNew = new ComputeBuffer((int)particleCount, 2 * sizeof(float));
        pressureField = new ComputeBuffer((int)particleCount, sizeof(float));
        pressureFieldNew = new ComputeBuffer((int)particleCount, sizeof(float));
        jacobiHelper = new ComputeBuffer((int)particleCount, 2 * sizeof(float));
        particleData = new ComputeBuffer((int)particleCount, 4 * sizeof(float));
        cbArray[0] = velocityField;
        cbArray[1] = velocityFieldNew;
        cbArray[2] = pressureField;
        cbArray[3] = pressureFieldNew;
        cbArray[4] = jacobiHelper;
        cbArray[5] = particleData;

        for (int i = 0; i < RT_COUNT; ++i)
        {
            
        }

        velocityField.SetData(new Vector2[particleCount]);
        velocityFieldNew.SetData(new Vector2[particleCount]);
        pressureField.SetData(new float[particleCount]);
        pressureFieldNew.SetData(new float[particleCount]);
        jacobiHelper.SetData(new Vector2[particleCount]);
        //particleData.SetData(new Vector2[particleCount]);

        velocityFieldBuffer = new Vector2[particleCount];
        particleDataBuffer = new Vector4[particleCount];

        finalTexture = new Texture2D((int)particleWidth, (int)particleWidth, TextureFormat.RGBAFloat, false);
        finalTexture.wrapMode = TextureWrapMode.Clamp;

        MaterialPropertyBlock mp = new MaterialPropertyBlock();
        mp.AddTexture(0, finalTexture);
        container.MySprite.SetPropertyBlock(mp);

        kernelIDs[0] = cShader.FindKernel("Advect");
        kernelIDs[1] = cShader.FindKernel("Diffuse");
        kernelIDs[2] = cShader.FindKernel("ApplyForces");
        kernelIDs[3] = cShader.FindKernel("Project");
        kernelIDs[4] = cShader.FindKernel("SwapOldToNew");
        kernelIDs[5] = cShader.FindKernel("SwapNewToOld");
        kernelIDs[6] = cShader.FindKernel("Initialize");

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.ADVECT], "VelocityField", velocityField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.ADVECT], "VelocityFieldNew", velocityFieldNew);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.DIFFUSE], "VelocityField", velocityFieldNew);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.DIFFUSE], "VelocityFieldNew", velocityField);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.APPLY_FORCE], "VelocityField", velocityField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.APPLY_FORCE], "VelocityFieldNew", velocityFieldNew);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.PROJECT], "VelocityField", velocityFieldNew);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.PROJECT], "VelocityFieldNew", velocityField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.PROJECT], "PressureField", pressureField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.PROJECT], "PressureFieldNew", pressureFieldNew);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.SWAP_TO_NEW], "VelocityField", velocityField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.SWAP_TO_NEW], "VelocityFieldNew", velocityFieldNew);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.SWAP_TO_OLD], "VelocityField", velocityFieldNew);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.SWAP_TO_OLD], "VelocityFieldNew", velocityField);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.INITIALIZE], "VelocityField", velocityFieldNew);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.INITIALIZE], "VelocityFieldNew", velocityField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.INITIALIZE], "PressureField", pressureField);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.INITIALIZE], "PressureFieldNew", pressureFieldNew);

        cShader.SetFloat("DeltaTime", Time.fixedDeltaTime);
        float dx = container.containerBase / (float)particleWidth;
        cShader.SetFloat("Dx", dx);
        cShader.SetInt("Width", (int)particleWidth);

        vfInitialized = true;
    }

    private void CalculateVectorField()
    {
        // setup particle data to texture
        for (uint i = 0; i < particleWidth; ++i )
        {
            for(uint j = 0; j < particleWidth; ++j)
            {
                uint flatCoord = i * particleWidth + j;

                particleDataBuffer[flatCoord].x = particles[flatCoord].transform.position.x;
                particleDataBuffer[flatCoord].y = particles[flatCoord].transform.position.y;
                particleDataBuffer[flatCoord].z = (float)particles[flatCoord].viscosity;
                particleDataBuffer[flatCoord].w = (float)particles[flatCoord].mass;
            }
        }
        particleData.SetData(particleDataBuffer);

        cShader.SetBuffer(kernelIDs[(int)KernelIDs.ADVECT], "ParticleData", particleData);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.DIFFUSE], "ParticleData", particleData);
        cShader.SetBuffer(kernelIDs[(int)KernelIDs.APPLY_FORCE], "ParticleData", particleData);

        // setup dropper
        cShader.SetFloats("DropperPosition", new float[] { dropper.CurrentForcePosition.x, dropper.CurrentForcePosition.y });
        cShader.SetFloats("DropperDirection", new float[] { dropper.CurrentForceDirection.x, dropper.CurrentForceDirection.y });
        cShader.SetFloat("DropperRadius", dropper.Radius);
        cShader.SetFloat("DropperForceValue", dropper.ForceValue);

        if(dropper.Active)
        {
            cShader.SetFloat("DropperForceMultiplier", 1.0f);
        }
        else
        {
            cShader.SetFloat("DropperForceMultiplier", 0.0f);
        }

        cShader.Dispatch(kernelIDs[(int)KernelIDs.ADVECT], (int)particleWidth / THREAD_COUNT, (int)particleWidth / THREAD_COUNT, 1);
        cShader.Dispatch(kernelIDs[(int)KernelIDs.SWAP_TO_OLD], (int)particleWidth / THREAD_COUNT, (int)particleWidth / THREAD_COUNT, 1);

        cShader.Dispatch(kernelIDs[(int)KernelIDs.APPLY_FORCE], (int)particleWidth / THREAD_COUNT, (int)particleWidth / THREAD_COUNT, 1);
        cShader.Dispatch(kernelIDs[(int)KernelIDs.SWAP_TO_OLD], (int)particleWidth / THREAD_COUNT, (int)particleWidth / THREAD_COUNT, 1);
        //cShader.Dispatch(kernelIDs[(int)KernelIDs.PROJECT], (int)particleWidth, (int)particleWidth, 1);

        ApplyTextureData(finalTexture, velocityField, velocityFieldBuffer);

        /*
        Profiler.BeginSample("Advect");
        Advect();
        Profiler.EndSample();
        Profiler.BeginSample("Diffuse");
        Diffuse();
        Profiler.EndSample();
        Profiler.BeginSample("ApplyForces");
        ApplyForces();
        Profiler.EndSample();
        Profiler.BeginSample("ComputePressure");
        ComputePressure();
        Profiler.EndSample();
        Profiler.BeginSample("SubtractPressureGradient");
        SubtractPressureGradient();
        Profiler.EndSample();
        Profiler.BeginSample("SolveBoundaries");
        SolveBoundaries();
        Profiler.EndSample();

        ApplyTextureData(ref vectorFieldTexture, velocityField);
        */
    }

    /*
    private void Advect()
    {
        for (uint i = 0; i < particleCount; ++i)
        {
            Vector2 cPos = Square1DCoords(i, particleWidth);
            Vector2 backPos = cPos - Time.fixedDeltaTime * new Vector2(velocityField[i].r, velocityField[i].g);

            backPos.x = Mathf.Clamp(backPos.x, 0.0f, (float)particleWidth - 1.0f);
            backPos.y = Mathf.Clamp(backPos.y, 0.0f, (float)particleWidth - 1.0f);

            Vector2 tl, tr, br, bl;
            tl = new Vector2(Mathf.Floor(backPos.x), Mathf.Ceil(backPos.y));
            tr = new Vector2(Mathf.Ceil(backPos.x), Mathf.Ceil(backPos.y));
            br = new Vector2(Mathf.Ceil(backPos.x), Mathf.Floor(backPos.y));
            bl = new Vector2(Mathf.Floor(backPos.x), Mathf.Floor(backPos.y));

            Color newVelocity =
                velocityField[Flatten2DCoords(tl, particleWidth)] +
                velocityField[Flatten2DCoords(tr, particleWidth)] +
                velocityField[Flatten2DCoords(br, particleWidth)] +
                velocityField[Flatten2DCoords(bl, particleWidth)];
            newVelocity /= 4.0f;

            velocityFieldNew[i] = newVelocity;
        }

        SwapColor(ref velocityField, ref velocityFieldNew);
    }

    private void Diffuse()
    {
        float alpha, rBeta;
        for (uint t = 0; t < JACOBI_ITERATIONS; ++t)
        {
            for (uint i = 0; i < particleCount; ++i)
            {
                alpha = (particles[i].radius * particles[i].radius * 4.0f) / ((float)particles[i].viscosity * Time.fixedDeltaTime);
                rBeta = 1.0f / (4.0f + alpha);

                Jacobi(i, out velocityFieldNew[i], alpha, rBeta, velocityField, velocityField[i], particleWidth);
            }
            SwapColor(ref velocityField, ref velocityFieldNew);
        }
    }

    // container elasticity will have to be taken into account here, I guess
    private void SolveBoundaries()
    {
        for (int i = 0; i < particleCount; ++i)
        {
            //left-right
            if (
                (i > 0 && i < particleWidth - 1) ||
                (i > (particleCount - particleWidth) && i != particleCount - 1)
                )
            {
                velocityField[i].r = 0.0f;
                velocityField[i].g = 0.0f;
                pressureField[i].r = 0.0f;
            }
            //top-down
            else if (
                (i % particleWidth == 0) ||
                (i % particleWidth == particleWidth - 1)
                )
            {
                velocityField[i].r = 0.0f;
                velocityField[i].g = 0.0f;
                pressureField[i].r = 0.0f;
            }

        }
    }

    private void ApplyForces()
    {
        if (dropper.Active)
        {
            Vector2 forceDir = dropper.CurrentForceDirection * dropper.ForceValue * Time.fixedDeltaTime;
            Vector2 forcePos = dropper.CurrentForcePosition;
            Vector2 vel = Vector2.zero;
            for (uint i = 0; i < particleCount; ++i)
            {
                float divisor = (Mathf.Pow(particles[i].transform.position.x - forcePos.x, 2.0f) +
                        Mathf.Pow(particles[i].transform.position.y - forcePos.y, 2.0f));
                if (divisor == 0.0f)
                    continue;
                vel = forceDir * Mathf.Exp(
                    dropper.Radius /
                    divisor
                    );

                velocityField[i].r = Mathf.Clamp(velocityField[i].r + vel.x * dropper.InsertedDensity, 0.0f, 1.0f);
                velocityField[i].g = Mathf.Clamp(velocityField[i].g + vel.y * dropper.InsertedDensity, 0.0f, 1.0f);

                // dunno why sometimes it spits out NaNs. This condition is for cleaning it up.
                if (velocityField[i].r != velocityField[i].r || velocityField[i].g != velocityField[i].g)
                {
                    velocityField[i] = Color.black;
                }
            }
            //Debug.Log(forcePos);
        }

    }

    private void ComputePressure()
    {
        float alpha;
        float rBeta = 0.25f;
        float halfrdx;
        Color div;
        for (uint t = 0; t < JACOBI_ITERATIONS; ++t)
        {
            for (uint i = 0; i < particleCount; ++i)
            {
                alpha = -(particles[i].radius * particles[i].radius * 4.0f);
                halfrdx = 1.0f / (2.0f * particles[i].radius * 2.0f);

                Divergence(i, out div, halfrdx, velocityField, particleWidth);
                Jacobi(i, out pressureFieldNew[i], alpha, rBeta, pressureField, div, particleWidth);
            }
            SwapColor(ref pressureField, ref pressureFieldNew);
        }
    }

    private void SubtractPressureGradient()
    {
        float halfrdx;
        for (uint i = 0; i < particleCount; ++i)
        {
            Vector2 coord2d = Square1DCoords(i, particleWidth);

            Color pL = pressureField[(int)(Mathf.Clamp(coord2d.x - 1.0f, 0.0f, (float)(particleWidth - 1)) * particleWidth + coord2d.y)];
            Color pR = pressureField[(int)(Mathf.Clamp(coord2d.x + 1.0f, 0.0f, (float)(particleWidth - 1)) * particleWidth + coord2d.y)];
            Color pB = pressureField[(int)(coord2d.x * particleWidth + Mathf.Clamp(coord2d.y - 1.0f, 0.0f, (float)(particleWidth - 1)))];
            Color pT = pressureField[(int)(coord2d.x * particleWidth + Mathf.Clamp(coord2d.y + 1.0f, 0.0f, (float)(particleWidth - 1)))];

            halfrdx = 1.0f / (2.0f * particles[i].radius * 2.0f);
            velocityField[i].r -= halfrdx * (pR.r - pL.r);
            velocityField[i].g -= halfrdx * (pT.r - pB.r);
        }
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

    private void Jacobi
        (
        uint coord,
        out Color xNew,
        float alpha,
        float rBeta,
        Color[] xField,
        Color b,
        uint width
        )
    {
        Vector2 coord2d = new Vector2(coord / width, coord % width);

        Color xL = xField[(int)(Mathf.Clamp(coord2d.x - 1.0f, 0.0f, (float)(width - 1)) * width + coord2d.y)];
        Color xR = xField[(int)(Mathf.Clamp(coord2d.x + 1.0f, 0.0f, (float)(width - 1)) * width + coord2d.y)];
        Color xB = xField[(int)(coord2d.x * width + Mathf.Clamp(coord2d.y - 1.0f, 0.0f, (float)(width - 1)))];
        Color xT = xField[(int)(coord2d.x * width + Mathf.Clamp(coord2d.y + 1.0f, 0.0f, (float)(width - 1)))];

        xNew = (xL + xR + xB + xT + alpha * b) * rBeta;

        // fixing alpha
        xNew.a = 1.0f;
    }

    private void Divergence
        (
        uint coord,
        out Color xNew,
        float halfrdx,
        Color[] xField,
        uint width
        )
    {
        Vector2 coord2d = Square1DCoords(coord, width);

        Color xL = xField[(int)(Mathf.Clamp(coord2d.x - 1.0f, 0.0f, (float)(width - 1)) * width + coord2d.y)];
        Color xR = xField[(int)(Mathf.Clamp(coord2d.x + 1.0f, 0.0f, (float)(width - 1)) * width + coord2d.y)];
        Color xB = xField[(int)(coord2d.x * width + Mathf.Clamp(coord2d.y - 1.0f, 0.0f, (float)(width - 1)))];
        Color xT = xField[(int)(coord2d.x * width + Mathf.Clamp(coord2d.y + 1.0f, 0.0f, (float)(width - 1)))];

        xNew = new Color(halfrdx * ((xR.r - xL.r) + (xT.g - xB.g)), 0.0f, 0.0f);

        // fixing alpha
        xNew.a = 1.0f;
    }

    private void ApplyTextureData(ref Texture2D tex, Color[] field)
    {
        tex.SetPixels(field);
        tex.Apply();
    }
     */

    private void ApplyTextureData(Texture2D tex, ComputeBuffer field, Vector2[] buffer)
    {
        field.GetData(buffer);
        for (uint i = 0; i < particleWidth; ++i )
        {
            for(uint j = 0; j < particleWidth; ++j)
            {
                uint id = i * particleWidth + j;
                tex.SetPixel((int)j, (int)i, new Color(buffer[id].x, buffer[id].y, 0.0f, 1.0f));
            }
        }
            
        tex.Apply();
    }

    private void ApplyVectorField()
    {

    }

    private void SwapTextureContent(ref Texture2D first, ref Texture2D second)
    {

    }

    private void SwapColorContent(ref Color[] first, ref Color[] second, uint length)
    {
        Color buffer;
        for (uint i = 0; i < length; ++i)
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
        DestroyParticles();
        particles = new FluidParticle[particleCount];

        float x = initialPosition.transform.position.x;
        float y = initialPosition.transform.position.y;

        particleWidth = (uint)Mathf.Sqrt(particleCount);

        for (uint i = 0; i < particleWidth; ++i)
        {
            for (uint j = 0; j < particleWidth; ++j)
            {
                particles[j + particleWidth * i] = (FluidParticle)Instantiate(baseObject, new Vector2(x, y), Quaternion.identity);
                particles[j + particleWidth * i].viscosity = baseObject.viscosity;
                particles[j + particleWidth * i].position = particles[j + particleWidth * i].transform.position;
                particles[j + particleWidth * i].radius = container.containerBase / (float)particleWidth / 2.0f;
                CalculatePosition(ref x, ref y, particleCount, false);
            }
            CalculatePosition(ref x, ref y, particleCount, true);
        }

        startSimulation = true;
    }

    public void CalculatePosition(ref float inputX, ref float inputY, uint count, bool moveUp)
    {
        switch (count)
        {
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
        if (moveUp)
        {
            inputX = initialPosition.transform.position.x;
            inputY += particleOffsetY;
        }
    }

    public void DestroyParticles()
    {
        if (particles[0] != null)
        {
            for (int i = 0; i < particles.Length; i++)
            {
                Destroy(particles[i].gameObject);
            }
        }
    }

    public void DestroyInteractiveObject(InteractiveObject io)
    {
        Debug.Log("Destroy");

        objects.Remove(io);
        Destroy(io.gameObject);

        canDelete = false;
    }

    #endregion
}
