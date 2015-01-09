package com.plodz.cartracker;

import java.util.ArrayList;
import java.util.Date;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.Map;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.KeyEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ExpandableListAdapter;
import android.widget.ExpandableListView;
import android.widget.ListView;
import android.widget.SimpleExpandableListAdapter;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.TextView.OnEditorActionListener;

import com.google.android.gms.common.*;

public class MainActivity extends Activity {
	
	public static DataSource data;
	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        checkForGooglePlay();
        data = new DataSource(this);
        initializeGlobals();
    }
    
    @Override
    protected void onStop()
    {
    	super.onStop();
    }
    
    @Override
    protected void onDestroy()
    {
    	super.onDestroy();

    	data.open();
    	data.saveFuels();
    	data.saveAllGlobals();
    	data.close();
    }
    
    protected void checkForGooglePlay()
    {
    	int result = GooglePlayServicesUtil.isGooglePlayServicesAvailable(this);
    	if(result != ConnectionResult.SUCCESS) finish();
    }
    
    protected void initializeGlobals()
    {
    	data.open();
    	boolean result;
    	result = data.loadFuelPrices();
    	if(!result) loadDefaultFuelPrices();
    	result = data.loadMiscGlobals();
    	if(!result) loadDefaultMiscGlobals();
    	data.close();
    }
    
    protected void loadDefaultMiscGlobals()
    {
    	Globals.myFuelConsumption = 6.0f;				
    	Globals.myFuelType = Globals.fuelType.ON;	
    	Globals.DBG_updateRatio = 0.00005f;
    	Globals.checkDelay = 1;
    	Globals.mapZoomMultiplier = 16.0f;
    	Globals.lastUpdate = new GregorianCalendar();
    }
    
    protected void loadDefaultFuelPrices()
    {
    	Globals.priceON = 0.0f;
    	Globals.priceLPG = 0.0f;
    	Globals.pricePB95 = 0.0f;
    	Globals.pricePB98 = 0.0f;
    }
    
    public void onStartButtonClick(View v)
    {
    	Intent intent = new Intent(this, TrackActivity.class);
    	startActivity(intent);
    }
    
    public void onLogButtonClick(View v)
    {
    	Intent intent = new Intent(this, LogActivity.class);
    	startActivity(intent);
    }
    
    public void onChartButtonClick(View v)
    {
    	Intent intent = new Intent(this, GlobalStatsActivity.class);
    	startActivity(intent);
    }
    
    public void onPricesButtonClick(View v)
    {
    	Intent intent = new Intent(this, FuelActivity.class);
    	startActivity(intent);
    }
    
    public void onSettingsButtonClick(View v)
    {
    	Intent intent = new Intent(this, SettingsActivity.class);
    	startActivity(intent);
    }
}
