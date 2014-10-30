package com.plodz.cartracker;

public class TrackingController {
	
	protected MapController map;
	protected TrackActivity activity;
	public TrackingController(TrackActivity activity)
	{
		this.activity = activity;
		map = new MapController(activity);
	}
}
