package com.plodz.cartracker;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

public class TripStatsFragment extends Fragment {
private static final String ARG_SECTION_NUMBER = "section_number";
	private LogStatisticsController lsController;

	public static TripStatsFragment newInstance(int sectionNumber) {
		TripStatsFragment fragment = new TripStatsFragment();
		Bundle args = new Bundle();
		args.putInt(ARG_SECTION_NUMBER, sectionNumber);
		fragment.setArguments(args);
		return fragment;
	}
	
	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		View rootView = (View) inflater.inflate(R.layout.fragment_logstats, container,
				false);
		
		return rootView;
	}
	
	@Override
	public void onStart()
	{
		super.onStart();
		lsController = new LogStatisticsController((LogActivity)this.getActivity());
		lsController.initializeDefaults();
	}
	
	public void loadTripData(TripModel tm) { lsController.loadTripData(tm); };
}
