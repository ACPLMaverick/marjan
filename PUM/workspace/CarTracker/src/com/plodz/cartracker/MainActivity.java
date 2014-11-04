package com.plodz.cartracker;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import com.google.android.gms.common.*;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkForGooglePlay();
    }
    
    @Override
    protected void onDestroy()
    {
    	super.onDestroy();
    }
    
    protected void checkForGooglePlay()
    {
    	int result = GooglePlayServicesUtil.isGooglePlayServicesAvailable(this);
    	if(result != ConnectionResult.SUCCESS) finish();
    }
    
    public void onStartButtonClick(View v)
    {
    	Intent intent = new Intent(this, TrackActivity.class);
    	startActivity(intent);
    }
}
