using UnityEngine;
using System.Collections.Generic;

public class SnakeHead : SnakeBody
{
    #region Enum

    public enum DirectionType
    {
        UP,
        DOWN,
        STOP,
        RIGHT,
        LEFT
    }

    #endregion

    #region Events

    public class SnakePositionChangedEvent : UnityEngine.Events.UnityEvent { }
    public SnakePositionChangedEvent SnakePositionChanged = new SnakePositionChangedEvent();

    #endregion

    #region Fields

    [SerializeField]
    protected GameObject _BodyPartPrefab;
    [SerializeField]
    protected float _StartSpeed = 1.0f;
    [SerializeField]
    protected uint _StartNumberOfParts = 2;
    [SerializeField]
    protected DirectionType _StartDirection = DirectionType.UP;

    #endregion

    #region Properties

    public float Speed { get; protected set; }
    public int PartsCount { get { return _allBodyParts.Count; } }
    public int LastCollisionID
    {
        get
        {
            if(!_lastBodyCollision)
            {
                return _realCollisionId;
            }
            else
            {
                SnakeBody body = _lastBodyCollision;
                _lastBodyCollision = null;
                return _allBodyParts.FindIndex(x => (x == body));
            }
        }
    }
    public List<SnakeBody> PartsBent
    {
        get
        {
            List<SnakeBody> bent = new List<SnakeBody>();
            bent.Add(this);
            if (_allBodyParts[0].Direction != this.Direction)
            {
                bent.Add(_allBodyParts[0]);
            }

            for (int i = 0; i < PartsCount - 1; ++i)
            {
                if(_allBodyParts[i].Direction != _allBodyParts[i + 1].Direction)
                {
                    bent.Add(_allBodyParts[i + 1]);
                }
            }

            return bent;
        }
    }

    #endregion

    #region Protected

    protected SnakeBody _lastBodyCollision = null;
    protected float _invSpeed { get { return 1.0f / Mathf.Max(Speed, 0.00001f); } }
    protected float _movementTimer = 0.0f;
    protected int _realCollisionId = -1;

    protected List<SnakeBody> _allBodyParts = new List<SnakeBody>();
    protected DirectionType _currentDirectionType;

    #endregion

    #region MonoBehaviours

    // Use this for initialization
    protected override void Start()
    {
        base.Start();

        _distanceSinceLastDirectionChange = _sizeWorld.x;
        Speed = _StartSpeed;
        _currentDirectionType = (DirectionType)(((int)_StartDirection + 2) % 4);
        if(_currentDirectionType == DirectionType.STOP)
        {
            ++_currentDirectionType;
        }

        // this is hard coded as it cannot be the other way
        Head = this;
        Previous = null;
    }

    // Update is called once per frame
    protected override void Update()
    {
        base.Update();
    }

    protected override void OnTriggerEnter2D(Collider2D coll)
    {
        if (coll.gameObject.CompareTag("fruit"))
        {
            Fruit fr = coll.gameObject.GetComponent<Fruit>();
            if (fr != null)
            {
                PickFruit(fr);
            }
        }
        //else if(coll.gameObject.CompareTag("snake") || coll.gameObject.CompareTag("head"))
        //{
        //    // do nothing
        //}
        else
        {
            Kill();
        }
    }

    #endregion

    #region Functions Public

    public static Vector2 DirectionTypeToDirection(DirectionType type)
    {
        Vector2 v = new Vector2(0.0f, 0.0f);

        switch (type)
        {
            case DirectionType.DOWN:
                v = Vector2.down;
                break;
            case DirectionType.LEFT:
                v = Vector2.left;
                break;
            case DirectionType.RIGHT:
                v = Vector2.right;
                break;
            case DirectionType.UP:
                v = Vector2.up;
                break;
            default:
                v = Vector2.zero;
                break;
        }

        return v;
    }

    public static DirectionType DirectionToDirectionType(Vector2 dir)
    {
        if (dir == Vector2.up)
        {
            return DirectionType.UP;
        }
        else if (dir == Vector2.right)
        {
            return DirectionType.RIGHT;
        }
        else if (dir == Vector2.down)
        {
            return DirectionType.DOWN;
        }
        else if (dir == Vector2.left)
        {
            return DirectionType.LEFT;
        }
        else
        {
            return DirectionType.STOP;
        }
    }
        /// <summary>
        /// This override of the base Initialize only calls the Head-specific Initialize(Player) function. You should not use it.
        /// </summary>
    public override void Initialize(Player player, SnakeHead head, SnakeBody next, SnakeBody prev, uint number)
    {
        Initialize(player);
    }

    public void Initialize(Player player)
    {
        Direction = Vector2.up;
        MyPlayer = player;
        GetComponent<SpriteRenderer>().color = MyPlayer.MyColor;

        if(_allBodyParts.Count != 0)
        {
            for(int i = 0; i < _allBodyParts.Count; ++i)
            {
                DestroyImmediate(_allBodyParts[i].gameObject, false);
            }
        }
        _allBodyParts.Clear();

        for(int i = 1; i <= _StartNumberOfParts; ++i)
        {
            _allBodyParts.Add(CreateNewPart(i));
        }

        for (int i = 0; i < _StartNumberOfParts; ++i)
        {
            SnakeBody next, prev;

            if(i == _StartNumberOfParts - 1)
            {
                next = null;
            }
            else
            {
                next = _allBodyParts[i + 1];
            }

            if(i == 0)
            {
                prev = this;
            }
            else
            {
                prev = _allBodyParts[i - 1];
            }

            _allBodyParts[i].Initialize(MyPlayer, this, next, prev, (uint)(i + 1));
        }

        Next = _allBodyParts[0];
        Previous = null;
        _initialized = true;
    }

    /**
     * This is called by Player as an equivalent to Update() to ensure synchronization.
     */
    public virtual void Tick()
    {
        if (_initialized)
        {
            if (_movementTimer >= _invSpeed)
            {
                ApplyMovement();
                _currentDirectionType = DirectionToDirectionType(Direction);

                for (int i = 0; i < _allBodyParts.Count; ++i)
                {
                    _allBodyParts[i].Move();
                }
                _movementTimer = 0.0f;

                SnakePositionChanged.Invoke();
            }
            else
            {
                _movementTimer += Time.deltaTime;
            }
        }
    }

    public override void Kill()
    {
        base.Kill();
        _realCollisionId = 0;
        MyPlayer.Lose();
    }

    public void DestroyBody()
    {
        for(int i = 0; i < _allBodyParts.Count; ++i)
        {
            Destroy(_allBodyParts[i].gameObject);
        }
    }

    public void AssignDirection(DirectionType dir)
    {
        if(/*_distanceSinceLastDirectionChange >= _sizeWorld.x &&*/     // assuming it is square
            (Mathf.Abs((int)dir - (int)_currentDirectionType) > 1))   
        {
            Direction = DirectionTypeToDirection(dir);
        }
        else if(dir == DirectionType.STOP)
        {
            _currentDirectionType = dir;
            Direction = Vector2.zero;
        }
    }

    public void BodyPartCleanup()
    {
        _allBodyParts.RemoveAll(x => x == null);
    }

    public Transform GetBodyPartPosition(int i)
    {
        return _allBodyParts[i].GetComponent<Transform>();
    }

    public void OnFruitCollected(float addition)
    {
        Speed += addition + 0.1f * Speed;

        BodyPartCleanup();

        SnakeBody tail = _allBodyParts[_allBodyParts.Count - 1];
        SnakeBody next = CreateNewPart(_allBodyParts.Count + 1);
        tail.Next = next;
        next.Initialize(MyPlayer, this, null, tail, (uint)_allBodyParts.Count + 1);
        _allBodyParts.Add(next);
    }

    public void SetPositionsAndDirectionsForAllParts(int allPartCount, Vector2[] positions, DirectionType[] directions)
    {
        // data check
        if(positions.Length != directions.Length)
        {
            Debug.LogWarning("SnakeHead: Positions.Length != Directions.Length");
        }

        // it is not necessary to take fruit collecting / speed / points into account
        // as they will automatically be collected when moved snake to given position
        
        // get ids of currently bent parts
        Vector2[] directionVectors = new Vector2[directions.Length];
        for(int i = 0; i < directionVectors.Length; ++i)
        {
            directionVectors[i] = DirectionTypeToDirection(directions[i]);
        }
        int[] bendsIDs = new int[positions.Length];

        for(int i = 1; i < positions.Length; ++i)
        {
            Vector2 startPos = positions[i - 1];
            Vector2 endPos = positions[i];
            float length = Vector2.Distance(startPos, endPos);
            int numParts = Mathf.RoundToInt(length / _sizeWorld.x);

            bendsIDs[i] = Mathf.Clamp(numParts + bendsIDs[i - 1], 0, _allBodyParts.Count); 
            //  not adding one because I need a part with different direction, not last with the same one AND I take into account that
            // allbodyparts doesnt contain head.
        }

        transform.position = positions[0];
        Direction = DirectionTypeToDirection(directions[0]);
        _movementTimer = 0.0f;

        int bpCount = _allBodyParts.Count;
        float governingPartAccumulator = _sizeWorld.x;

        SnakeBody bentPart = this;
        for (int i = 0, b = 1; i < bpCount; ++i)
        {

            if (b < bendsIDs.Length /* for situations when only head is a bent part */ &&  (i + 1) == bendsIDs[b])
            {
                // this is a bent part. Set position and direction from received data
                // And set this as governing part
                _allBodyParts[i].transform.position = positions[b];
                _allBodyParts[i].Direction = directionVectors[b];

                ++b;
                bentPart = _allBodyParts[i];
                governingPartAccumulator = _sizeWorld.x;
            }
            else
            {
                // this is not a bent part, so just set it up after bent one
                _allBodyParts[i].transform.position = bentPart.transform.position - new Vector3(bentPart.Direction.x, bentPart.Direction.y, 0.0f) * governingPartAccumulator;
                _allBodyParts[i].Direction = bentPart.Direction;
                governingPartAccumulator += _sizeWorld.x;
            }
        }
    }

    public void RegisterCollision(SnakeBody body)
    {
        _lastBodyCollision = body;
    }

    #endregion

    #region Functions Protected

    protected SnakeBody CreateNewPart(int iter)
    {
        GameObject obj = Instantiate(_BodyPartPrefab);
        obj.name = obj.name + string.Format("_{0}", iter);
        obj.GetComponent<SpriteRenderer>().color = MyPlayer.MyColor;
        SnakeBody b = obj.GetComponent<SnakeBody>();
        return b;
    }

    #endregion
}
