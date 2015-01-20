package com.plodz.cartracker;

import java.util.ArrayList;
import java.util.Locale;

import com.google.android.gms.games.multiplayer.turnbased.TurnBasedMultiplayer.InitiateMatchResult;

import android.support.v7.app.ActionBarActivity;
import android.support.v7.app.ActionBar;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentTransaction;
import android.support.v4.app.FragmentPagerAdapter;
import android.app.Activity;
import android.os.Bundle;
import android.support.v4.view.ViewPager;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.webkit.WebView.FindListener;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemClickListener;
import android.widget.AdapterView.OnItemLongClickListener;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.TextView;

public class LogActivity extends FragmentActivity {

	/**
	 * The {@link android.support.v4.view.PagerAdapter} that will provide
	 * fragments for each of the sections. We use a {@link FragmentPagerAdapter}
	 * derivative, which will keep every loaded fragment in memory. If this
	 * becomes too memory intensive, it may be best to switch to a
	 * {@link android.support.v4.app.FragmentStatePagerAdapter}.
	 */
	SectionsPagerAdapter mSectionsPagerAdapter;

	/**
	 * The {@link ViewPager} that will host the section contents.
	 */
	private ViewPager mViewPager;
	public ArrayList<Fragment> myFragments;
	private ArrayList<TripModel> tmList;
	private ArrayAdapter<String> adapter;
	
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_log);
		
		myFragments = new ArrayList<Fragment>();
		myFragments.add(TripLogFragment.newInstance(0));
		myFragments.add(TripStatsFragment.newInstance(1));

		mSectionsPagerAdapter = new SectionsPagerAdapter(
				getSupportFragmentManager(), myFragments);

		// Set up the ViewPager with the sections adapter.
		mViewPager = (ViewPager) findViewById(R.id.pager);
		mViewPager.setAdapter(mSectionsPagerAdapter);
		mSectionsPagerAdapter.notifyDataSetChanged();
	}
	
	@Override
	protected void onStart()
	{
		super.onStart();
		
	}
	
	public void initializeListView()
	{
		MainActivity.data.open();
    	this.tmList = MainActivity.data.getAllTripModels();
    	MainActivity.data.close();
    	
    	ArrayList<String> tms = new ArrayList<String>();
    	for(TripModel mod: tmList)
    	{
    		System.out.println(mod.toString());
    		tms.add(mod.toString());
    	}
    	
    	ListView lv = ((TripLogFragment)myFragments.get(0)).listView;
    	adapter = new ArrayAdapter<String>(this, R.layout.settings_listviewlayout, tms);
    	if(lv != null) lv.setAdapter(adapter);
    	adapter.notifyDataSetChanged();
    	
    	lv.setOnItemClickListener(new OnItemClickListener(){

			@Override
			public void onItemClick(AdapterView<?> arg0, View arg1, int arg2,
					long arg3) {
				mViewPager.setCurrentItem(1, true);
				TripStatsFragment tf = (TripStatsFragment)myFragments.get(1);
				tf.loadTripData(tmList.get(arg2));
			}
    		
    	});
    	
    	lv.setOnItemLongClickListener(new OnItemLongClickListener() {

			@Override
			public boolean onItemLongClick(AdapterView<?> arg0, View arg1,
					int arg2, long arg3) {
				long remTime = tmList.get(arg2).getStartTime();
				tmList.remove(arg2);
				MainActivity.data.open();
				MainActivity.data.removeTripAt(remTime);
				MainActivity.data.close();
				
				initializeListView();
				return true;
			}
    		
		});
	}
	
	

	/**
	 * A {@link FragmentPagerAdapter} that returns a fragment corresponding to
	 * one of the sections/tabs/pages.
	 */
	public class SectionsPagerAdapter extends FragmentPagerAdapter {
		
		private ArrayList<Fragment> list;
		public static final int pos = 0;
		
		public SectionsPagerAdapter(FragmentManager fm, ArrayList<Fragment> list) {
			super(fm);
			this.list = list;
		}

		@Override
		public Fragment getItem(int position) {
			// getItem is called to instantiate the fragment for the given page.
			// Return a PlaceholderFragment (defined as a static inner class
			// below).
			return list.get(position);
		}

		@Override
		public int getCount() {
			return list.size();
		}

		@Override
		public CharSequence getPageTitle(int position) {
			Locale l = Locale.getDefault();
			switch (position) {
			case 0:
				return getString(R.string.title_section1).toUpperCase(l);
			case 1:
				return getString(R.string.title_section2).toUpperCase(l);
			}
			return null;
		}
		
		public int getPos() {
		    return pos;
		}

		 public void setPos(int pos) {
			 //this.pos = pos;
		 }
	}
}
