#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.3),
    on 五月 30, 2024, at 14:39
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware, parallel
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.3'
expName = 'test1'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\0cifar\\cifar10_dog_test_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = False
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "hello" ---
    Hello = visual.TextStim(win=win, name='Hello',
        text='Hello! This is a test from VLIS LAB.\n\nWe will get your EEG signals when you look at the selected images. Please stay focused.\n\nThank you for your efforts!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "part1" ---
    Part1 = visual.TextStim(win=win, name='Part1',
        text='Look and read',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "textBlock1" ---
    text = visual.TextStim(win=win, name='text',
        text='Dog',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    readText = visual.TextStim(win=win, name='readText',
        text='Please read the word "Dog".',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    fixation2 = visual.TextStim(win=win, name='fixation2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    p_port = parallel.ParallelPort(address='0x5FF8')
    
    # --- Initialize components for Routine "quicklooktext" ---
    quickLookText = visual.TextStim(win=win, name='quickLookText',
        text='take a glance at the text',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "quickTextBlock1" ---
    quickText = visual.TextStim(win=win, name='quickText',
        text='Dog',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    p_port2 = parallel.ParallelPort(address='0x5FF8')
    quickfixation = visual.TextStim(win=win, name='quickfixation',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "quicklooktext" ---
    quickLookText = visual.TextStim(win=win, name='quickLookText',
        text='take a glance at the text',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "quickTextBlock2" ---
    quickText_2 = visual.TextStim(win=win, name='quickText_2',
        text='Dog',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    p_port2_2 = parallel.ParallelPort(address='0x5FF8')
    quickfixation_2 = visual.TextStim(win=win, name='quickfixation_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Part3" ---
    quicklookImage = visual.TextStim(win=win, name='quicklookImage',
        text='look at the image and think',
        font='Arial',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "imageBlock1" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(1, 1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    p_port3 = parallel.ParallelPort(address='0x5FF8')
    fixation_2 = visual.TextStim(win=win, name='fixation_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "quickTextBlock2" ---
    quickText_2 = visual.TextStim(win=win, name='quickText_2',
        text='Dog',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    p_port2_2 = parallel.ParallelPort(address='0x5FF8')
    quickfixation_2 = visual.TextStim(win=win, name='quickfixation_2',
        text='+',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "hello" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('hello.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    helloComponents = [Hello]
    for thisComponent in helloComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "hello" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Hello* updates
        
        # if Hello is starting this frame...
        if Hello.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Hello.frameNStart = frameN  # exact frame index
            Hello.tStart = t  # local t and not account for scr refresh
            Hello.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Hello, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Hello.started')
            # update status
            Hello.status = STARTED
            Hello.setAutoDraw(True)
        
        # if Hello is active this frame...
        if Hello.status == STARTED:
            # update params
            pass
        
        # if Hello is stopping this frame...
        if Hello.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Hello.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                Hello.tStop = t  # not accounting for scr refresh
                Hello.tStopRefresh = tThisFlipGlobal  # on global time
                Hello.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Hello.stopped')
                # update status
                Hello.status = FINISHED
                Hello.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in helloComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hello" ---
    for thisComponent in helloComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('hello.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "part1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('part1.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    part1Components = [Part1]
    for thisComponent in part1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "part1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Part1* updates
        
        # if Part1 is starting this frame...
        if Part1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Part1.frameNStart = frameN  # exact frame index
            Part1.tStart = t  # local t and not account for scr refresh
            Part1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Part1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Part1.started')
            # update status
            Part1.status = STARTED
            Part1.setAutoDraw(True)
        
        # if Part1 is active this frame...
        if Part1.status == STARTED:
            # update params
            pass
        
        # if Part1 is stopping this frame...
        if Part1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > Part1.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                Part1.tStop = t  # not accounting for scr refresh
                Part1.tStopRefresh = tThisFlipGlobal  # on global time
                Part1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Part1.stopped')
                # update status
                Part1.status = FINISHED
                Part1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in part1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "part1" ---
    for thisComponent in part1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('part1.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=2.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "textBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('textBlock1.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        textBlock1Components = [text, fixation, readText, fixation2, p_port]
        for thisComponent in textBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "textBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *readText* updates
            
            # if readText is starting this frame...
            if readText.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
                # keep track of start time/frame for later
                readText.frameNStart = frameN  # exact frame index
                readText.tStart = t  # local t and not account for scr refresh
                readText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(readText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'readText.started')
                # update status
                readText.status = STARTED
                readText.setAutoDraw(True)
            
            # if readText is active this frame...
            if readText.status == STARTED:
                # update params
                pass
            
            # if readText is stopping this frame...
            if readText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > readText.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    readText.tStop = t  # not accounting for scr refresh
                    readText.tStopRefresh = tThisFlipGlobal  # on global time
                    readText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'readText.stopped')
                    # update status
                    readText.status = FINISHED
                    readText.setAutoDraw(False)
            
            # *fixation2* updates
            
            # if fixation2 is starting this frame...
            if fixation2.status == NOT_STARTED and tThisFlip >= 5.0-frameTolerance:
                # keep track of start time/frame for later
                fixation2.frameNStart = frameN  # exact frame index
                fixation2.tStart = t  # local t and not account for scr refresh
                fixation2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation2.started')
                # update status
                fixation2.status = STARTED
                fixation2.setAutoDraw(True)
            
            # if fixation2 is active this frame...
            if fixation2.status == STARTED:
                # update params
                pass
            
            # if fixation2 is stopping this frame...
            if fixation2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation2.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation2.tStop = t  # not accounting for scr refresh
                    fixation2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation2.stopped')
                    # update status
                    fixation2.status = FINISHED
                    fixation2.setAutoDraw(False)
            # *p_port* updates
            
            # if p_port is starting this frame...
            if p_port.status == NOT_STARTED and text.status==STARTED:
                # keep track of start time/frame for later
                p_port.frameNStart = frameN  # exact frame index
                p_port.tStart = t  # local t and not account for scr refresh
                p_port.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port.started', t)
                # update status
                p_port.status = STARTED
                p_port.status = STARTED
                win.callOnFlip(p_port.setData, int(1))
            
            # if p_port is stopping this frame...
            if p_port.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port.tStop = t  # not accounting for scr refresh
                    p_port.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port.stopped', t)
                    # update status
                    p_port.status = FINISHED
                    win.callOnFlip(p_port.setData, int(2))
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in textBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "textBlock1" ---
        for thisComponent in textBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('textBlock1.stopped', globalClock.getTime(format='float'))
        if p_port.status == STARTED:
            win.callOnFlip(p_port.setData, int(2))
        # the Routine "textBlock1" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 2.0 repeats of 'trials'
    
    
    # --- Prepare to start Routine "quicklooktext" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('quicklooktext.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    quicklooktextComponents = [quickLookText]
    for thisComponent in quicklooktextComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "quicklooktext" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *quickLookText* updates
        
        # if quickLookText is starting this frame...
        if quickLookText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            quickLookText.frameNStart = frameN  # exact frame index
            quickLookText.tStart = t  # local t and not account for scr refresh
            quickLookText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(quickLookText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'quickLookText.started')
            # update status
            quickLookText.status = STARTED
            quickLookText.setAutoDraw(True)
        
        # if quickLookText is active this frame...
        if quickLookText.status == STARTED:
            # update params
            pass
        
        # if quickLookText is stopping this frame...
        if quickLookText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > quickLookText.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                quickLookText.tStop = t  # not accounting for scr refresh
                quickLookText.tStopRefresh = tThisFlipGlobal  # on global time
                quickLookText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickLookText.stopped')
                # update status
                quickLookText.status = FINISHED
                quickLookText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in quicklooktextComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "quicklooktext" ---
    for thisComponent in quicklooktextComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('quicklooktext.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=5.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            globals()[paramName] = thisTrial_2[paramName]
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                globals()[paramName] = thisTrial_2[paramName]
        
        # --- Prepare to start Routine "quickTextBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('quickTextBlock1.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        quickTextBlock1Components = [quickText, p_port2, quickfixation]
        for thisComponent in quickTextBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "quickTextBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.30000000000000004:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *quickText* updates
            
            # if quickText is starting this frame...
            if quickText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                quickText.frameNStart = frameN  # exact frame index
                quickText.tStart = t  # local t and not account for scr refresh
                quickText.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickText, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickText.started')
                # update status
                quickText.status = STARTED
                quickText.setAutoDraw(True)
            
            # if quickText is active this frame...
            if quickText.status == STARTED:
                # update params
                pass
            
            # if quickText is stopping this frame...
            if quickText.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickText.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    quickText.tStop = t  # not accounting for scr refresh
                    quickText.tStopRefresh = tThisFlipGlobal  # on global time
                    quickText.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickText.stopped')
                    # update status
                    quickText.status = FINISHED
                    quickText.setAutoDraw(False)
            # *p_port2* updates
            
            # if p_port2 is starting this frame...
            if p_port2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port2.frameNStart = frameN  # exact frame index
                p_port2.tStart = t  # local t and not account for scr refresh
                p_port2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port2.started', t)
                # update status
                p_port2.status = STARTED
                p_port2.status = STARTED
                win.callOnFlip(p_port2.setData, int(10))
            
            # if p_port2 is stopping this frame...
            if p_port2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port2.tStartRefresh + 0.2-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port2.tStop = t  # not accounting for scr refresh
                    p_port2.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port2.stopped', t)
                    # update status
                    p_port2.status = FINISHED
                    win.callOnFlip(p_port2.setData, int(11))
            
            # *quickfixation* updates
            
            # if quickfixation is starting this frame...
            if quickfixation.status == NOT_STARTED and tThisFlip >= 0.2-frameTolerance:
                # keep track of start time/frame for later
                quickfixation.frameNStart = frameN  # exact frame index
                quickfixation.tStart = t  # local t and not account for scr refresh
                quickfixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickfixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickfixation.started')
                # update status
                quickfixation.status = STARTED
                quickfixation.setAutoDraw(True)
            
            # if quickfixation is active this frame...
            if quickfixation.status == STARTED:
                # update params
                pass
            
            # if quickfixation is stopping this frame...
            if quickfixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickfixation.tStartRefresh + 0.1-frameTolerance:
                    # keep track of stop time/frame for later
                    quickfixation.tStop = t  # not accounting for scr refresh
                    quickfixation.tStopRefresh = tThisFlipGlobal  # on global time
                    quickfixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickfixation.stopped')
                    # update status
                    quickfixation.status = FINISHED
                    quickfixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in quickTextBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "quickTextBlock1" ---
        for thisComponent in quickTextBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('quickTextBlock1.stopped', globalClock.getTime(format='float'))
        if p_port2.status == STARTED:
            win.callOnFlip(p_port2.setData, int(11))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.300000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 5.0 repeats of 'trials_2'
    
    
    # --- Prepare to start Routine "quicklooktext" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('quicklooktext.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    quicklooktextComponents = [quickLookText]
    for thisComponent in quicklooktextComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "quicklooktext" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *quickLookText* updates
        
        # if quickLookText is starting this frame...
        if quickLookText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            quickLookText.frameNStart = frameN  # exact frame index
            quickLookText.tStart = t  # local t and not account for scr refresh
            quickLookText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(quickLookText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'quickLookText.started')
            # update status
            quickLookText.status = STARTED
            quickLookText.setAutoDraw(True)
        
        # if quickLookText is active this frame...
        if quickLookText.status == STARTED:
            # update params
            pass
        
        # if quickLookText is stopping this frame...
        if quickLookText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > quickLookText.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                quickLookText.tStop = t  # not accounting for scr refresh
                quickLookText.tStopRefresh = tThisFlipGlobal  # on global time
                quickLookText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickLookText.stopped')
                # update status
                quickLookText.status = FINISHED
                quickLookText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in quicklooktextComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "quicklooktext" ---
    for thisComponent in quicklooktextComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('quicklooktext.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=10.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='trials_3')
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "quickTextBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('quickTextBlock2.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        quickTextBlock2Components = [quickText_2, p_port2_2, quickfixation_2]
        for thisComponent in quickTextBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "quickTextBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *quickText_2* updates
            
            # if quickText_2 is starting this frame...
            if quickText_2.status == NOT_STARTED and tThisFlip >= 0.00-frameTolerance:
                # keep track of start time/frame for later
                quickText_2.frameNStart = frameN  # exact frame index
                quickText_2.tStart = t  # local t and not account for scr refresh
                quickText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickText_2.started')
                # update status
                quickText_2.status = STARTED
                quickText_2.setAutoDraw(True)
            
            # if quickText_2 is active this frame...
            if quickText_2.status == STARTED:
                # update params
                pass
            
            # if quickText_2 is stopping this frame...
            if quickText_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickText_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    quickText_2.tStop = t  # not accounting for scr refresh
                    quickText_2.tStopRefresh = tThisFlipGlobal  # on global time
                    quickText_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickText_2.stopped')
                    # update status
                    quickText_2.status = FINISHED
                    quickText_2.setAutoDraw(False)
            # *p_port2_2* updates
            
            # if p_port2_2 is starting this frame...
            if p_port2_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port2_2.frameNStart = frameN  # exact frame index
                p_port2_2.tStart = t  # local t and not account for scr refresh
                p_port2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port2_2.started', t)
                # update status
                p_port2_2.status = STARTED
                p_port2_2.status = STARTED
                win.callOnFlip(p_port2_2.setData, int(12))
            
            # if p_port2_2 is stopping this frame...
            if p_port2_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port2_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port2_2.tStop = t  # not accounting for scr refresh
                    p_port2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port2_2.stopped', t)
                    # update status
                    p_port2_2.status = FINISHED
                    win.callOnFlip(p_port2_2.setData, int(13))
            
            # *quickfixation_2* updates
            
            # if quickfixation_2 is starting this frame...
            if quickfixation_2.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                # keep track of start time/frame for later
                quickfixation_2.frameNStart = frameN  # exact frame index
                quickfixation_2.tStart = t  # local t and not account for scr refresh
                quickfixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickfixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickfixation_2.started')
                # update status
                quickfixation_2.status = STARTED
                quickfixation_2.setAutoDraw(True)
            
            # if quickfixation_2 is active this frame...
            if quickfixation_2.status == STARTED:
                # update params
                pass
            
            # if quickfixation_2 is stopping this frame...
            if quickfixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickfixation_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    quickfixation_2.tStop = t  # not accounting for scr refresh
                    quickfixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    quickfixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickfixation_2.stopped')
                    # update status
                    quickfixation_2.status = FINISHED
                    quickfixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in quickTextBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "quickTextBlock2" ---
        for thisComponent in quickTextBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('quickTextBlock2.stopped', globalClock.getTime(format='float'))
        if p_port2_2.status == STARTED:
            win.callOnFlip(p_port2_2.setData, int(13))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 10.0 repeats of 'trials_3'
    
    
    # --- Prepare to start Routine "Part3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('Part3.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    Part3Components = [quicklookImage]
    for thisComponent in Part3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Part3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *quicklookImage* updates
        
        # if quicklookImage is starting this frame...
        if quicklookImage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            quicklookImage.frameNStart = frameN  # exact frame index
            quicklookImage.tStart = t  # local t and not account for scr refresh
            quicklookImage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(quicklookImage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'quicklookImage.started')
            # update status
            quicklookImage.status = STARTED
            quicklookImage.setAutoDraw(True)
        
        # if quicklookImage is active this frame...
        if quicklookImage.status == STARTED:
            # update params
            pass
        
        # if quicklookImage is stopping this frame...
        if quicklookImage.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > quicklookImage.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                quicklookImage.tStop = t  # not accounting for scr refresh
                quicklookImage.tStopRefresh = tThisFlipGlobal  # on global time
                quicklookImage.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quicklookImage.stopped')
                # update status
                quicklookImage.status = FINISHED
                quicklookImage.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Part3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Part3" ---
    for thisComponent in Part3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('Part3.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials_4 = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('cifar10png/dog_test.xlsx'),
        seed=None, name='trials_4')
    thisExp.addLoop(trials_4)  # add the loop to the experiment
    thisTrial_4 = trials_4.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
    if thisTrial_4 != None:
        for paramName in thisTrial_4:
            globals()[paramName] = thisTrial_4[paramName]
    
    for thisTrial_4 in trials_4:
        currentLoop = trials_4
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_4.rgb)
        if thisTrial_4 != None:
            for paramName in thisTrial_4:
                globals()[paramName] = thisTrial_4[paramName]
        
        # --- Prepare to start Routine "imageBlock1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('imageBlock1.started', globalClock.getTime(format='float'))
        image.setImage(Pic)
        # keep track of which components have finished
        imageBlock1Components = [image, p_port3, fixation_2]
        for thisComponent in imageBlock1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "imageBlock1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image* updates
            
            # if image is starting this frame...
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image.started')
                # update status
                image.status = STARTED
                image.setAutoDraw(True)
            
            # if image is active this frame...
            if image.status == STARTED:
                # update params
                pass
            
            # if image is stopping this frame...
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.tStopRefresh = tThisFlipGlobal  # on global time
                    image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.stopped')
                    # update status
                    image.status = FINISHED
                    image.setAutoDraw(False)
            # *p_port3* updates
            
            # if p_port3 is starting this frame...
            if p_port3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port3.frameNStart = frameN  # exact frame index
                p_port3.tStart = t  # local t and not account for scr refresh
                p_port3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port3.started', t)
                # update status
                p_port3.status = STARTED
                p_port3.status = STARTED
                win.callOnFlip(p_port3.setData, int(100))
            
            # if p_port3 is stopping this frame...
            if p_port3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port3.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port3.tStop = t  # not accounting for scr refresh
                    p_port3.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port3.stopped', t)
                    # update status
                    p_port3.status = FINISHED
                    win.callOnFlip(p_port3.setData, int(101))
            
            # *fixation_2* updates
            
            # if fixation_2 is starting this frame...
            if fixation_2.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                # keep track of start time/frame for later
                fixation_2.frameNStart = frameN  # exact frame index
                fixation_2.tStart = t  # local t and not account for scr refresh
                fixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_2.started')
                # update status
                fixation_2.status = STARTED
                fixation_2.setAutoDraw(True)
            
            # if fixation_2 is active this frame...
            if fixation_2.status == STARTED:
                # update params
                pass
            
            # if fixation_2 is stopping this frame...
            if fixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_2.tStop = t  # not accounting for scr refresh
                    fixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_2.stopped')
                    # update status
                    fixation_2.status = FINISHED
                    fixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in imageBlock1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "imageBlock1" ---
        for thisComponent in imageBlock1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('imageBlock1.stopped', globalClock.getTime(format='float'))
        if p_port3.status == STARTED:
            win.callOnFlip(p_port3.setData, int(101))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        
        # --- Prepare to start Routine "quickTextBlock2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('quickTextBlock2.started', globalClock.getTime(format='float'))
        # keep track of which components have finished
        quickTextBlock2Components = [quickText_2, p_port2_2, quickfixation_2]
        for thisComponent in quickTextBlock2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "quickTextBlock2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.1:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *quickText_2* updates
            
            # if quickText_2 is starting this frame...
            if quickText_2.status == NOT_STARTED and tThisFlip >= 0.00-frameTolerance:
                # keep track of start time/frame for later
                quickText_2.frameNStart = frameN  # exact frame index
                quickText_2.tStart = t  # local t and not account for scr refresh
                quickText_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickText_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickText_2.started')
                # update status
                quickText_2.status = STARTED
                quickText_2.setAutoDraw(True)
            
            # if quickText_2 is active this frame...
            if quickText_2.status == STARTED:
                # update params
                pass
            
            # if quickText_2 is stopping this frame...
            if quickText_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickText_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    quickText_2.tStop = t  # not accounting for scr refresh
                    quickText_2.tStopRefresh = tThisFlipGlobal  # on global time
                    quickText_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickText_2.stopped')
                    # update status
                    quickText_2.status = FINISHED
                    quickText_2.setAutoDraw(False)
            # *p_port2_2* updates
            
            # if p_port2_2 is starting this frame...
            if p_port2_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                p_port2_2.frameNStart = frameN  # exact frame index
                p_port2_2.tStart = t  # local t and not account for scr refresh
                p_port2_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(p_port2_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('p_port2_2.started', t)
                # update status
                p_port2_2.status = STARTED
                p_port2_2.status = STARTED
                win.callOnFlip(p_port2_2.setData, int(12))
            
            # if p_port2_2 is stopping this frame...
            if p_port2_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > p_port2_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    p_port2_2.tStop = t  # not accounting for scr refresh
                    p_port2_2.tStopRefresh = tThisFlipGlobal  # on global time
                    p_port2_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('p_port2_2.stopped', t)
                    # update status
                    p_port2_2.status = FINISHED
                    win.callOnFlip(p_port2_2.setData, int(13))
            
            # *quickfixation_2* updates
            
            # if quickfixation_2 is starting this frame...
            if quickfixation_2.status == NOT_STARTED and tThisFlip >= 0.05-frameTolerance:
                # keep track of start time/frame for later
                quickfixation_2.frameNStart = frameN  # exact frame index
                quickfixation_2.tStart = t  # local t and not account for scr refresh
                quickfixation_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(quickfixation_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'quickfixation_2.started')
                # update status
                quickfixation_2.status = STARTED
                quickfixation_2.setAutoDraw(True)
            
            # if quickfixation_2 is active this frame...
            if quickfixation_2.status == STARTED:
                # update params
                pass
            
            # if quickfixation_2 is stopping this frame...
            if quickfixation_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > quickfixation_2.tStartRefresh + 0.05-frameTolerance:
                    # keep track of stop time/frame for later
                    quickfixation_2.tStop = t  # not accounting for scr refresh
                    quickfixation_2.tStopRefresh = tThisFlipGlobal  # on global time
                    quickfixation_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'quickfixation_2.stopped')
                    # update status
                    quickfixation_2.status = FINISHED
                    quickfixation_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in quickTextBlock2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "quickTextBlock2" ---
        for thisComponent in quickTextBlock2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('quickTextBlock2.stopped', globalClock.getTime(format='float'))
        if p_port2_2.status == STARTED:
            win.callOnFlip(p_port2_2.setData, int(13))
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.100000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_4'
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
