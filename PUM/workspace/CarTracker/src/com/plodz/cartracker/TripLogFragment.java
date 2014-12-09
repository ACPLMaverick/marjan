package com.plodz.cartracker;

import java.util.ArrayList;
import android.support.v4.app.Fragment;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.ListView;
import android.widget.RelativeLayout;

public class TripLogFragment extends Fragment {
	private static final String ARG_SECTION_NUMBER = "section_number";
	public ListView listView;
	public static TripLogFragment newInstance(int sectionNumber) {
		TripLogFragment fragment = new TripLogFragment();
		Bundle args = new Bundle();
		args.putInt(ARG_SECTION_NUMBER, sectionNumber);
		fragment.setArguments(args);
		return fragment;
	}
	
	@Override
	public View onCreateView(LayoutInflater inflater, ViewGroup container,
			Bundle savedInstanceState) {
		View rootView = (View) inflater.inflate(R.layout.fragment_log, container,
				false);
		listView = (ListView) ((RelativeLayout)rootView).getChildAt(0);
		//initializeListView();
		((LogActivity)this.getActivity()).initializeListView();
		return rootView;
	}
}
