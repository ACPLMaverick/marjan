using UnityEngine;
using System.Collections.Generic;

public class GameController : MonoBehaviour
{
    #region Fields

    [SerializeField]
    protected List<Player> _Players;

    [SerializeField]
    protected List<GameObject> _FruitPrefabs;

    [SerializeField]
    protected int _MaxFruitsOnLevel = 3;

    [SerializeField]
    protected float _FruitGenerateDelayMin = 1.0f;

    [SerializeField]
    protected float _FruitGenerateDelayMax = 6.0f;

    [SerializeField]
    protected int _PointsToWin = 0;

    [SerializeField]
    protected int _LengthToWin = 0;


    #endregion

    #region Properties

    public GameController Instance { get; private set; }

    #endregion

    #region Protected

    protected Transform _fruitAreaMin;
    protected Transform _fruitAreaMax;

    protected List<Player> _playersInGame = new List<Player>();
    protected List<Fruit> _fruitsOnLevel = new List<Fruit>();
    protected float _currentDelay = 0.0f;
    protected float _delayTimer = 0.0f;

    #endregion

    #region MonoBehaviours

    private void Awake()
    {
        Instance = this;
        _fruitAreaMin = GetComponentsInChildren<Transform>()[1];
        _fruitAreaMax = GetComponentsInChildren<Transform>()[2];
    }

    // Use this for initialization
    void Start()
    {
        for(int i = 0; i < _Players.Count; ++i)
        {
            _Players[i].EventLose.AddListener(new UnityEngine.Events.UnityAction<Player>(OnPlayerLose));
            _playersInGame.Add(_Players[i]);
        }
    }

    // Update is called once per frame
    void Update()
    {
        // Fruit generation.
        if(_fruitsOnLevel.Count < _MaxFruitsOnLevel && _FruitPrefabs.Count > 0)
        {
            if(_delayTimer <= 0.0f) // generate new fruit now
            {
                GenerateNewFruit();
                _currentDelay = Random.Range(_FruitGenerateDelayMin, _FruitGenerateDelayMax);
                _delayTimer = _currentDelay;
            }
            else // decrement the timer
            {
                _delayTimer -= Time.deltaTime;
            }
        }
    }

    #endregion

    #region Functions Public

    #endregion

    #region Functions Protected

    protected void OnPlayerLose(Player player)
    {
        Debug.Log(player.gameObject.name + ", you died, sir.");
        _playersInGame.Remove(player);

        if(_playersInGame.Count == 1)
        {
            WinForPlayer(_playersInGame[0]);
        }
        else if(_playersInGame.Count == 0)
        {
            Draw();
        }
    }

    protected void OnFruitCollected(Fruit fruit)
    {
        _fruitsOnLevel.Remove(fruit);
        _currentDelay = Random.Range(_FruitGenerateDelayMin, _FruitGenerateDelayMax);
        _delayTimer = _currentDelay;
    }

    protected void WinForPlayer(Player player)
    {
        player.Stop();
        Debug.Log(player.gameObject.name + ", you won, sir.");
        Application.Quit();
    }

    protected void Draw()
    {
        Debug.Log("It's a draw!");
        Application.Quit();
    }

    protected void GenerateNewFruit()
    {
        int n = Random.Range(0, _FruitPrefabs.Count - 1);
        GameObject newFruitObject = Instantiate(_FruitPrefabs[n]);
        newFruitObject.GetComponent<Transform>().position = new Vector3
            (
                Random.Range(_fruitAreaMin.position.x, _fruitAreaMax.position.x),
                Random.Range(_fruitAreaMin.position.y, _fruitAreaMax.position.y),
                Random.Range(_fruitAreaMin.position.z, _fruitAreaMax.position.z)
            );
        Fruit fr = newFruitObject.GetComponent<Fruit>();
        fr.EventCollected.AddListener(new UnityEngine.Events.UnityAction<Fruit>(OnFruitCollected));
        _fruitsOnLevel.Add(fr);
    }

    #endregion
}
