using UnityEngine;
using UnityEngine.Events;
using System.Collections;

/// <summary>
/// Fruit is an element which makes player longer and faster
/// </summary>
public class Fruit : MonoBehaviour
{
    #region Fields

    [SerializeField]
    protected int _Points = 1;

    [SerializeField]
    protected float _SpeedAddition = 0.5f;

    #endregion

    #region Events

    public class UnityEventFruitCollected : UnityEvent<Fruit> { }
    public UnityEventFruitCollected EventCollected = new UnityEventFruitCollected();

    #endregion

    #region Properties

    public int Points { get { return _Points; } }
    public float SpeedAddition { get { return _SpeedAddition; } }

    #endregion

    #region Protected

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    void OnTriggerEnter2D()
    {
        EventCollected.Invoke(this);
        Kill();
    }

    #endregion

    #region Functions Public

    #endregion

    #region Functions Protected
    /// <summary>
    /// Destroys fruit on collect
    /// </summary>
    protected void Kill()
    {
        Destroy(gameObject);
    }

    #endregion
}
