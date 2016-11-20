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

    #endregion

    #region Protected

    protected float _invSpeed { get { return 1.0f / Mathf.Max(Speed, 0.00001f); } }
    protected float _movementTimer = 0.0f;

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
        AssignDirection(_StartDirection);

        // this is hard coded as it cannot be the other way
        Head = this;
        Previous = null;
    }

    // Update is called once per frame
    protected override void Update()
    {
        base.Update();

        if(_initialized)
        {
            if (_movementTimer >= _invSpeed)
            {
                ApplyMovement();
                if(Direction == Vector2.up)
                {
                    _currentDirectionType = DirectionType.UP;
                }
                else if(Direction == Vector2.right)
                {
                    _currentDirectionType = DirectionType.RIGHT;
                }
                else if(Direction == Vector2.down)
                {
                    _currentDirectionType = DirectionType.DOWN;
                }
                else if(Direction == Vector2.left)
                {
                    _currentDirectionType = DirectionType.LEFT;
                }

                for (int i = 0; i < _allBodyParts.Count; ++i)
                {
                    _allBodyParts[i].Move();
                }
                _movementTimer = 0.0f;
            }
            else
            {
                _movementTimer += Time.deltaTime;
            }
        }
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
        else if(coll.gameObject.CompareTag("snake") || coll.gameObject.CompareTag("head"))
        {
            // do nothing
        }
        else
        {
            Kill();
        }
    }

    #endregion

        #region Functions Public

        /// <summary>
        /// This override of the base Initialize only calls the Head-specific Initialize(Player) function. You should not use it.
        /// </summary>
    public override void Initialize(Player player, SnakeHead head, SnakeBody next, SnakeBody prev, uint number)
    {
        Initialize(player);
    }

    public void Initialize(Player player)
    {
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

    public override void Kill()
    {
        base.Kill();
        MyPlayer.Lose();
    }

    public void AssignDirection(DirectionType dir)
    {
        if(/*_distanceSinceLastDirectionChange >= _sizeWorld.x &&*/     // assuming it is square
            (Mathf.Abs((int)dir - (int)_currentDirectionType) > 1))   
        {
            switch (dir)
            {
                case DirectionType.DOWN:
                    Direction = Vector2.down;
                    break;
                case DirectionType.LEFT:
                    Direction = Vector2.left;
                    break;
                case DirectionType.RIGHT:
                    Direction = Vector2.right;
                    break;
                case DirectionType.UP:
                    Direction = Vector2.up;
                    break;
                default:
                    Direction = Vector2.zero;
                    break;
            }
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
