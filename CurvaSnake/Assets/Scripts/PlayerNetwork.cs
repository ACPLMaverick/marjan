using UnityEngine;
using System.Collections;
using Network;

/// <summary>
/// Derived class for player set as network player
/// </summary>
public class PlayerNetwork : Player
{

    #region MonoBehaviours

    // Use this for initialization
    protected override void Start()
    {
        base.Start();
    }

    // Update is called once per frame
    protected override void Update()
    {
        base.Update();
    }

    #endregion

    #region Functions Public

    public override void Initialize(int id)
    {
        base.Initialize(id);
    }

    public override void UpdateFromPlayerData(PlayerData data)
    {
        base.UpdateFromPlayerData(data);

        // sets all body parts positions according to player data
        ApplyDirectlyPlayerData(data);
    }

    #endregion
}
