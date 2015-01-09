package com.plodz.cartracker;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.GregorianCalendar;

import org.achartengine.ChartFactory;
import org.achartengine.GraphicalView;
import org.achartengine.model.TimeSeries;
import org.achartengine.model.XYMultipleSeriesDataset;
import org.achartengine.model.XYSeries;
import org.achartengine.renderer.SimpleSeriesRenderer;
import org.achartengine.renderer.XYMultipleSeriesRenderer;
import org.achartengine.renderer.XYSeriesRenderer;

import android.content.Intent;
import android.graphics.Color;
import android.util.DisplayMetrics;
import android.util.TypedValue;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.TableLayout.LayoutParams;

public class GlobalStatsController {

	private final int chartLineWidth = 3;
	private final int chartBackgroundColor = Color.WHITE;
	private final int chartMarginsColor = Color.DKGRAY;
	private final int chartLabelsColor = Color.WHITE;
	private final int chartYLabelPadding = 20;
	private final int chartAxisTitleTextSize = 18;
	private final int chartTitleTextSize = 25;
	private final int chartLabelsTextSize = 12;
	private final int chartMarginTop = 45;
	private final int chartMarginLeft = 50;
	private final int chartMarginBottom = 15;
	private final int chartMarginRight = 15;
	
	private final int chartCount = 5;
	
	private final int[] chartLineColors = {Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.MAGENTA};
	
	private GlobalStatsActivity activity;
	private DataSource data;
	
	private ArrayList<Trip> trips;
	private ArrayList<Trip> tripsFiltered;
	
	private GregorianCalendar filterStartDate;
	private GregorianCalendar filterEndDate;
	private String filterStartAddress;
	private String filterEndAddress;
	
	private TextView tvGStotaltripcount;
	private TextView tvGStotalfuelcost;
	private TextView tvGSavgfuelcost;
	private TextView tvGStotalfuelcons;
	private TextView tvGSavgfuelcons;
	private TextView tvGSavgspeed;
	private TextView tvGSavgdistance;
	private TextView tvGStotaldistance;
	private TextView tvGSavgtime;
	private TextView tvGStotaltime;
	
	private int tripCount = 0;
	private float totalCost = 0.0f;
	private float avgCost = 0.0f;
	private float totalCons = 0.0f;
	private float avgCons = 0.0f;
	private float avgSpeed = 0.0f;
	private float avgDist = 0.0f;
	private float totalDist = 0.0f;
	private long avgTime = 0;
	private long totalTime = 0;
	
	GregorianCalendar calAvgTime;
	GregorianCalendar calTotalTime;
	
	GraphicalView gvCost;
	GraphicalView gvCons;
	GraphicalView gvDist;
	GraphicalView gvAvgSpd;
	GraphicalView gvTime;
	
	public GlobalStatsController(GlobalStatsActivity activity, DataSource data)
	{
		this.activity = activity;
		this.data = data;
		this.trips = new ArrayList<Trip>();
		
		this.tripsFiltered = new ArrayList<Trip>();
		this.filterStartDate = new GregorianCalendar();
		this.filterStartDate.setTimeInMillis(0);
		this.filterEndDate = new GregorianCalendar();
		this.filterEndDate.setTimeInMillis(Long.MAX_VALUE);
		this.filterStartAddress = "";
		this.filterEndAddress = "";
		
		this.calAvgTime = new GregorianCalendar();
		this.calTotalTime = new GregorianCalendar();
	}
	
	public void initialize()
	{
		tvGStotaltripcount = (TextView) activity.findViewById(R.id.tvGStotaltripcount);
		tvGStotalfuelcost = (TextView) activity.findViewById(R.id.tvGStotalfuelcost);
		tvGStotalfuelcons = (TextView) activity.findViewById(R.id.tvGStotalfuelconsumed);
		tvGSavgfuelcost = (TextView) activity.findViewById(R.id.tvGSavgfuelcost);
		tvGSavgfuelcons = (TextView) activity.findViewById(R.id.tvGSavgfuelconsumed);
		tvGSavgspeed = (TextView) activity.findViewById(R.id.tvGSavgspeed);
		tvGSavgdistance = (TextView) activity.findViewById(R.id.tvGSavgdistance);
		tvGStotaldistance = (TextView) activity.findViewById(R.id.tvGStotaldistance);
		tvGSavgtime = (TextView) activity.findViewById(R.id.tvGSavgtime);
		tvGStotaltime = (TextView) activity.findViewById(R.id.tvGStotaltime);
		
		data.open();
		ArrayList<TripModel> tms = data.getAllTripModels();
		data.close();
		
		if(tms.size() > 0)
		{
			for(TripModel tm : tms)
			{
				trips.add(new Trip(tm));
			}
		}
		
		updateFilters();
		updateData();
		if(tripsFiltered.size() > 0)
		{
			initializeCharts();
			updateCharts();
		}
	}
	
	public void updateData()
	{
		tripCount = tripsFiltered.size();
		totalCost = 0.0f;
		avgCost = 0.0f;
		totalCons = 0.0f;
		avgCons = 0.0f;
		avgSpeed = 0.0f;
		avgDist = 0.0f;
		totalDist = 0.0f;
		avgTime = 0;
		totalTime = 0;
		
		for(Trip trip : tripsFiltered)
		{
			totalCost += trip.fuelCost;
			avgCost += trip.fuelCost;
			totalCons += trip.fuelConsumed;
			avgCons += trip.fuelConsumed;
			avgSpeed += trip.avgSpeed;
			avgDist += trip.distance;
			totalDist += trip.distance;
			avgTime += (trip.endTime.getTimeInMillis() - trip.startTime.getTimeInMillis());
			totalTime += (trip.endTime.getTimeInMillis() - trip.startTime.getTimeInMillis());
		}
		
		if(tripsFiltered.size() != 0)
		{
			avgCost /= tripsFiltered.size();
			avgCons /= tripsFiltered.size();
			avgSpeed /= tripsFiltered.size();
			avgDist /= tripsFiltered.size();
			avgTime /= tripsFiltered.size();
		}
		
//		long odjemnik = 24*3600000;
//		avgTime -= odjemnik;
//		totalTime -= odjemnik;
		
		calAvgTime.setTimeInMillis(avgTime);
		calTotalTime.setTimeInMillis(totalTime);
		
		updateUI();
	}
	
	public void updateFilters()
	{
		// read new filter data from UI - TBA
		
		tripsFiltered.clear();
		
		for(Trip trip : trips)
		{
			boolean ifAdd = false;
			if(trip.startTime.getTimeInMillis() >= this.filterStartDate.getTimeInMillis() &&
					trip.endTime.getTimeInMillis() <= this.filterEndDate.getTimeInMillis())
			{
				ifAdd = true;
			}
			else ifAdd = false;
			
			if(this.filterStartAddress != "" && ifAdd)
			{
				if(trip.startAddress.equals(this.filterStartAddress)) ifAdd = true;
				else ifAdd = false;
			}
			
			if(this.filterEndAddress != "" && ifAdd)
			{
				if(trip.endAddress.equals(this.filterEndAddress)) ifAdd = true;
				else ifAdd = false;
			}
			
			if(ifAdd) tripsFiltered.add(trip);
		}
	}
	
	private void initializeCharts()
	{
		for(int i = 0; i < chartCount; i++)
		{
			initializeChart(i);
		}
	}
	
	private void initializeChart(int selection)
	{
		XYMultipleSeriesRenderer renderer = new XYMultipleSeriesRenderer();
		XYSeriesRenderer rnd = new XYSeriesRenderer();
		
		DisplayMetrics metrics = activity.getResources().getDisplayMetrics();
				
		rnd.setLineWidth(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartLineWidth, metrics));
		rnd.setColor(chartLineColors[selection % chartCount]);
		
	    renderer.setXTitle(activity.getString(R.string.str_chart_val_date));
	    
	    renderer.setXAxisMax(tripsFiltered.get(tripsFiltered.size() - 1).startTime.getTimeInMillis());
	    renderer.setXAxisMin(tripsFiltered.get(0).startTime.getTimeInMillis());
		
	    renderer.setBackgroundColor(chartBackgroundColor);
		renderer.setApplyBackgroundColor(true);
		renderer.setMarginsColor(chartMarginsColor);
		renderer.setXLabelsColor(chartLabelsColor);
		renderer.setYLabelsColor(0, chartLabelsColor);
		
		renderer.setYLabelsPadding(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartYLabelPadding, metrics));
	    
	    renderer.setAxisTitleTextSize(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartAxisTitleTextSize, metrics));
	    renderer.setChartTitleTextSize(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartTitleTextSize, metrics));
	    renderer.setLabelsTextSize(TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartLabelsTextSize, metrics));
	    
	    renderer.setMargins(new int[] { 
	    		(int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartMarginTop, metrics), 
	    		(int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartMarginLeft, metrics), 
	    		(int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartMarginBottom, metrics), 
	    		(int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, chartMarginRight, metrics) });
	    
	    
	    renderer.setShowLegend(false);
	    renderer.setShowLabels(true);
	    renderer.setInScroll(true);
	    renderer.setPanEnabled(false, false);
	    
	    switch(selection)
		{
	    case 1:
	    	renderer.setChartTitle(activity.getString(R.string.str_chart_cons));
	    	renderer.setYTitle(activity.getString(R.string.str_stat_consumedVal));
	    	break;
	    case 2:
	    	renderer.setChartTitle(activity.getString(R.string.str_chart_dist));
	    	renderer.setYTitle(activity.getString(R.string.str_stat_distanceVal));
	    	break;
	    case 3:
	    	renderer.setChartTitle(activity.getString(R.string.str_chart_spd));
	    	renderer.setYTitle(activity.getString(R.string.str_stat_speedAvgVal));
	    	break;
	    case 4:
	    	renderer.setChartTitle(activity.getString(R.string.str_chart_time));
	    	renderer.setYTitle(activity.getString(R.string.str_stat_timeCurrVal));
	    	break;
	    default:
	    	renderer.setChartTitle(activity.getString(R.string.str_chart_cost));
	    	renderer.setYTitle(activity.getString(R.string.str_stat_costVal));
	    	break;
		}
	    
	    renderer.addSeriesRenderer(rnd);
		
		XYMultipleSeriesDataset dsCost = new XYMultipleSeriesDataset();
		TimeSeries sCost = new TimeSeries("");
		
		for(Trip trip : tripsFiltered)
		{
			switch(selection)
			{
		    case 1:
		    	sCost.add(trip.startTime.getTime(), trip.fuelConsumed);
		    	break;
		    case 2:
		    	sCost.add(trip.startTime.getTime(), trip.distance);
		    	break;
		    case 3:
		    	sCost.add(trip.startTime.getTime(), trip.avgSpeed);
		    	break;
		    case 4:
		    	sCost.add(trip.startTime.getTime(), (trip.endTime.getTimeInMillis() - trip.startTime.getTimeInMillis()));
		    	break;
		    default:
		    	sCost.add(trip.startTime.getTime(), trip.fuelCost);
		    	break;
			}
		}
		
		dsCost.addSeries(sCost);
		
		String dateFormat = "MM-dd HH:mm";
		if((tripsFiltered.get(tripsFiltered.size() - 1).startTime.getTimeInMillis() - 
				tripsFiltered.get(0).startTime.getTimeInMillis()) < 86400000)
		{
			dateFormat = "HH:mm:ss";
		}
		
		TableLayout trCost;
		
		switch(selection)
		{
	    case 1:
	    	trCost = (TableLayout) activity.findViewById(R.id.gs_chartsLayoutCons);
	    	gvCons = ChartFactory.getTimeChartView(activity, dsCost, renderer, dateFormat);
			trCost.addView(gvCons, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
			trCost.invalidate();
	    	break;
	    case 2:
	    	trCost = (TableLayout) activity.findViewById(R.id.gs_chartsLayoutDist);
	    	gvDist = ChartFactory.getTimeChartView(activity, dsCost, renderer, dateFormat);
			trCost.addView(gvDist, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
			trCost.invalidate();
	    	break;
	    case 3:
	    	trCost = (TableLayout) activity.findViewById(R.id.gs_chartsLayoutAvgSpd);
	    	gvAvgSpd = ChartFactory.getTimeChartView(activity, dsCost, renderer, dateFormat);
			trCost.addView(gvAvgSpd, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
			trCost.invalidate();
	    	break;
	    case 4:
	    	trCost = (TableLayout) activity.findViewById(R.id.gs_chartsLayoutTime);
	    	gvTime = ChartFactory.getTimeChartView(activity, dsCost, renderer, dateFormat);
			trCost.addView(gvTime, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
			trCost.invalidate();
	    	break;
	    default:
	    	trCost = (TableLayout) activity.findViewById(R.id.gs_chartsLayoutCost);
	    	gvCost = ChartFactory.getTimeChartView(activity, dsCost, renderer, dateFormat);
			trCost.addView(gvCost, new LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT));
			trCost.invalidate();
	    	break;
		}
	}
	
	private void updateUI()
	{
		tvGStotaltripcount.setText(String.valueOf(this.tripCount));
		tvGStotalfuelcost.setText(String.format("%.2f", this.totalCost) + activity.getString(R.string.str_stat_costVal));
		tvGSavgfuelcost.setText(String.format("%.2f", this.avgCost) + activity.getString(R.string.str_stat_costVal));
		tvGStotalfuelcons.setText(String.format("%.2f", this.totalCons) + activity.getString(R.string.str_stat_consumedVal));
		tvGSavgfuelcons.setText(String.format("%.2f", this.avgCons) + activity.getString(R.string.str_stat_consumedVal));
		tvGSavgspeed.setText(String.format("%.2f", this.avgSpeed) + activity.getString(R.string.str_stat_speedAvgVal));
		tvGSavgdistance.setText(String.format("%.2f", this.avgDist) + activity.getString(R.string.str_stat_distanceVal));
		tvGStotaldistance.setText(String.format("%.2f", this.totalDist) + activity.getString(R.string.str_stat_distanceVal));

		String avgTimeStr, totalTimeStr;
		
		avgTimeStr = timeToString(avgTime);
		totalTimeStr = timeToString(totalTime);
		
		tvGSavgtime.setText(avgTimeStr);
		tvGStotaltime.setText(totalTimeStr);
	}
	
	private void updateCharts()
	{
		
	}
	
	private String timeToString(long time)
	{
		String toRet = "";
		
		int sec, min, h, d;
		time /= 1000;
		sec = (int) time % 60;
		time /= 60;
		min = (int) time % 60;
		time /= 60;
		h = (int) time % 60;
		time /= 24;
		d = (int) time;
		
		String sd, sh, sm, ss;
		sd = String.valueOf(d);
		sh = String.valueOf(h);
		sm = String.valueOf(min);
		ss = String.valueOf(sec);
		
		if(sh.length() < 2) sh = "0" + sh;
		if(sm.length() < 2) sm = "0" + sm;
		if(ss.length() < 2) ss = "0" + ss;
		
		toRet = sd + 
				" " + activity.getString(R.string.str_gs_days) + ", " +
				sh + ":" + 
				sm + ":" + 
				ss;
		
		return toRet;
	}
}
