using UnityEngine;
using System.Collections;

public class ParticleController : MonoBehaviour {

    private Vector3 startPos;
    private ParticleSystem particleSys;

	// Use this for initialization
	void Start () {
        startPos = this.transform.position;
        particleSys = this.GetComponent<ParticleSystem>();
	}
	
	// Update is called once per frame
	void Update () {
	
	}

    public void FireParticle(Vector3 position)
    {
        this.transform.position = position;
        particleSys.Emit(60);
    }
}
