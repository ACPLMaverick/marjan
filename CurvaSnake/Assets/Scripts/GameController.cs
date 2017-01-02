using UnityEngine;
using System;
using System.Collections.Generic;

public class GameController : MonoBehaviour
{
    #region Enums

    public enum NetworkMode
    {
        Server,
        Client
    }

    #endregion

    #region Fields

    [SerializeField]
    protected GameObject _ServerPrefab;

    [SerializeField]
    protected GameObject _ClientPrefab;

    [SerializeField]
    protected string _ServerAddress;

    [SerializeField]
    protected NetworkMode _NetworkMode;

    [SerializeField]
    protected PlayerLocal _LocalPlayer;

    [SerializeField]
    protected List<Player> _Players = new List<Player>();

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

    [SerializeField]
    protected PlayerSpawner[] _Spawners;

    [SerializeField]
    protected PlayerNetwork _PlayerNetworkPrefab;

    #endregion

    #region Properties

    public static GameController Instance { get; private set; }
    public static float TimeSeconds { get; protected set; }

    public PlayerSpawner[] Spawners { get; private set; }

    #endregion

    #region Protected

    protected Network.Server _localServer;

     /** 
     * Every game has its own network client. 
     * Here a server login is performed, ID is obtained from server
     * And listening on game state change is performed.
     */
    protected Network.Client _gameClient;

    protected List<Network.Client> _networkTestClients = new List<Network.Client>();

    //protected Transform _fruitAreaMin;
    //protected Transform _fruitAreaMax;

    protected List<Player> _playersInGame = new List<Player>();
    protected List<Fruit> _fruitsOnLevel = new List<Fruit>();
    protected List<int> _connectedPlayerIds = new List<int>();
    protected List<int> _disconnectedPlayerIds = new List<int>();
    protected List<KeyValuePair<int, Network.PlayerData>> _playerDatasToUpdate = new List<KeyValuePair<int, Network.PlayerData>>();
    //protected float _currentDelay = 0.0f;
    //protected float _delayTimer = 0.0f;
    protected int _localPlayerIDToSpawn = -1;
    protected bool _canEnableLocalPlayer = true;

    #endregion

    #region MonoBehaviours

    private void Awake()
    {
        Instance = this;
        //_fruitAreaMin = GetComponentsInChildren<Transform>()[1];
        //_fruitAreaMax = GetComponentsInChildren<Transform>()[2];
    }

    // Use this for initialization
    void Start()
    {
        if(_NetworkMode == NetworkMode.Server)
        {
            GameObject srv = Instantiate(_ServerPrefab);
            srv.transform.parent = transform;
            _localServer = srv.GetComponent<Network.Server>();
        }

        _gameClient = Instantiate(_ClientPrefab).GetComponent<Network.Client>();
        _gameClient.gameObject.transform.parent = transform;
        _gameClient.SetServerAddress(_ServerAddress);
        _gameClient.Connect(CallbackOnClientConnected);

        //network players for tetin
        //foreach(Player pl in _Players)
        //{
        //    Network.Client cl = Instantiate(_ClientPrefab).GetComponent<Network.Client>();
        //    cl.gameObject.transform.parent = transform;
        //    cl.SetServerAddress(_ServerAddress);
        //    cl.Connect(CallbackOnClientConnected);
        //    _networkTestClients.Add(cl);
        //    _playersInGame.Add(pl);

        //    cl.EventPlayerConnected.AddListener(CallbackOnAnotherPlayerConnected);
        //    cl.EventPlayerDisconnected.AddListener(CallbackOnAnotherPlayerDisconnected);
        //    cl.EventPlayerDataReceived.AddListener(CallbackOnClientDataReceived);

        //    pl.EventLose.AddListener(new UnityEngine.Events.UnityAction<Player>(OnPlayerLose));
        //    pl.Initialize(2);
        //}

        _Players.Add(_LocalPlayer);
        _LocalPlayer.gameObject.SetActive(false);

        _localServer.EventAddApple.AddListener(GenerateNewFruit);

    }

    // Update is called once per frame
    void Update()
    {
        // Fruit generation.
        /*
        if(_fruitsOnLevel.Count < _MaxFruitsOnLevel && _FruitPrefabs.Count > 0)
        {
            if(_delayTimer <= 0.0f) // generate new fruit now
            {
                GenerateNewFruit();
                _currentDelay = UnityEngine.Random.Range(_FruitGenerateDelayMin, _FruitGenerateDelayMax);
                _delayTimer = _currentDelay;
            }
            else // decrement the timer
            {
                _delayTimer -= Time.deltaTime;
            }
        }
        */
        if(_canEnableLocalPlayer && !_LocalPlayer.enabled)
        {
            _LocalPlayer.enabled = true;
        }

        if(_localPlayerIDToSpawn != -1)
        {
            OnClientConnected(_localPlayerIDToSpawn);
            _localPlayerIDToSpawn = -1;
        }

        if(_connectedPlayerIds.Count != 0)
        {
            for(int i = 0; i < _connectedPlayerIds.Count; ++i)
            {
                //OnClientConnected(_connectedPlayerIds[i]);
                SpawnNetworkPlayer(_connectedPlayerIds[i]);
            }
            _connectedPlayerIds.Clear();
        }

        if (_disconnectedPlayerIds.Count != 0)
        {
            for (int i = 0; i < _connectedPlayerIds.Count; ++i)
            {
                DestroyNetworkPlayer(_disconnectedPlayerIds[i]);
            }
            _disconnectedPlayerIds.Clear();
        }
        if(_playerDatasToUpdate.Count != 0)
        {
            for(int i = 0; i < _playerDatasToUpdate.Count; ++i)
            {
                OnClientDataRecieved(_playerDatasToUpdate[i].Key, _playerDatasToUpdate[i].Value);
            }
            _playerDatasToUpdate.Clear();
        }

        //FOR TETING
        if(Input.GetKeyDown(KeyCode.Space))
        {
            SpawnNetworkPlayer(2);
        }
        //

        TimeSeconds = Time.time;
    }

    #endregion

    #region Functions Public

    public PlayerSpawner GetSpawner(int i)
    {
        return _Spawners[Mathf.Clamp(i - 1, 0, _Spawners.Length - 1)];
    }

    public void SpawnNetworkPlayer(int id)
    {
        if (!GetSpawner(id).IsPlayerAssigned)
        {
            PlayerNetwork netPlayer = Instantiate<PlayerNetwork>(_PlayerNetworkPrefab);

            netPlayer.EventLose.AddListener(new UnityEngine.Events.UnityAction<Player>(OnPlayerLose));
            netPlayer.Initialize(id);
            _playersInGame.Add(netPlayer);

            netPlayer.gameObject.SetActive(true);
        }
        else
        {
            Debug.LogFormat("Player {0} already logged in", id);
        }
    }

    public void DestroyNetworkPlayer(int id)
    {
        int pCount = _playersInGame.Count;
        for(int i = 0; i < pCount; ++i)
        {
            if(_playersInGame[i].MyID == id)
            {
                GetSpawner(id).IsPlayerAssigned = false;
                Destroy(_playersInGame[i].gameObject);
                _playersInGame.RemoveAt(i);
                return;
            }
        }

        if (_playersInGame.Count == 1)
        {
            WinForPlayer(_playersInGame[0]);
        }
    }

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
        //_currentDelay = UnityEngine.Random.Range(_FruitGenerateDelayMin, _FruitGenerateDelayMax);
        //_delayTimer = _currentDelay;
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

    protected void GenerateNewFruit(Vector2 pos)
    {
        int n = UnityEngine.Random.Range(0, _FruitPrefabs.Count - 1);
        GameObject newFruitObject = Instantiate(_FruitPrefabs[n]);
        newFruitObject.GetComponent<Transform>().position = pos;//new Vector3
 //           (
 //               UnityEngine.Random.Range(_fruitAreaMin.position.x, _fruitAreaMax.position.x),
 //               UnityEngine.Random.Range(_fruitAreaMin.position.y, _fruitAreaMax.position.y),
 //               UnityEngine.Random.Range(_fruitAreaMin.position.z, _fruitAreaMax.position.z)
 //           );
        Fruit fr = newFruitObject.GetComponent<Fruit>();
        fr.EventCollected.AddListener(new UnityEngine.Events.UnityAction<Fruit>(OnFruitCollected));
        _fruitsOnLevel.Add(fr);
    }

    #region NetworkRelated

    protected void CallbackOnClientConnected(int id)
    {
        _localPlayerIDToSpawn = id;
    }

    protected void OnClientConnected(int id)
    {
        _gameClient.EventPlayerConnected.AddListener(CallbackOnAnotherPlayerConnected);
        _gameClient.EventPlayerDisconnected.AddListener(CallbackOnAnotherPlayerDisconnected);
        _gameClient.EventPlayerDataReceived.AddListener(CallbackOnClientDataReceived);
        _gameClient.EventAddApple.AddListener(CallbackOnAddApple);

        _LocalPlayer.EventLose.AddListener(new UnityEngine.Events.UnityAction<Player>(OnPlayerLose));
        _LocalPlayer.Initialize(id, _gameClient);
        _playersInGame.Add(_LocalPlayer);

        _LocalPlayer.gameObject.SetActive(true);
    }

    protected void CallbackOnClientDataReceived(int playerID, Network.PlayerData data)
    {
        _playerDatasToUpdate.Add(new KeyValuePair<int, Network.PlayerData>(playerID, data));
    }

    protected void OnClientDataRecieved(int playerID, Network.PlayerData data)
    {
        int playerCount = _playersInGame.Count;
        for (int i = 0; i < playerCount; ++i)
        {
            if (_playersInGame[i].MyID == playerID)
            {
                _playersInGame[i].UpdateFromPlayerData(data);
                break;
            }

            //_playersInGame[i].UpdateFromPlayerData(data);
        }
    }

    protected void CallbackOnAnotherPlayerConnected(int id)
    {
        _connectedPlayerIds.Add(id);
    }

    protected void CallbackOnAnotherPlayerDisconnected(int id)
    {
        _disconnectedPlayerIds.Add(id);
    }

    protected void CallbackOnAddApple(Vector2 pos)
    {
        GenerateNewFruit(pos);
    }


    #endregion

    #endregion
}
