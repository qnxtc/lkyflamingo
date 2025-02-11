import os
import queue
import sys
import time

import numpy as np
import pandas as pd

###############################################
from agent.flamingo.SA_ClientAgent import SA_ClientAgent as Client
from message.Message import MessageType
from util.util import log_print


class Kernel:

    def __init__(self, kernel_name, random_state=None):
        # kernel_name is for human readers only.初始化内核的属性
        self.name = kernel_name
        self.random_state = random_state

        if not random_state:
            raise ValueError("A valid, seeded np.random.RandomState object is required " +
                             "for the Kernel", self.name)
            sys.exit()
        # 1. 内核名称和随机状态的初始化

        # A single message queue to keep everything organized by increasing
        # delivery timestamp.
        self.messages = queue.PriorityQueue()
        # 2. 消息队列的初始化
        # self.messages = queue.PriorityQueue()：初始化一个优先队列PriorityQueue并赋值给self.messages。
        # 优先队列的特点是元素会按照优先级进行排序，在这个模拟内核中，元素（即消息）会按照交付时间戳（delivery timestamp）进行排序，
        # 交付时间早的消息会排在前面，这样可以保证消息按照时间顺序依次处理。

        # currentTime is None until after kernelStarting() event completes
        # for all agents.  This is a pd.Timestamp that includes the date.
        self.currentTime = None
        # 3. 当前时间的初始化
        # self.currentTime = None：将内核对象的currentTime属性初始化为None。
        # 在所有代理的kernelStarting()事件完成之前，这个属性的值会一直保持为None。
        # currentTime是一个pandas的Timestamp对象，包含日期信息，用于表示当前模拟的时间。

        # Timestamp at which the Kernel was created.  Primarily used to
        # create a unique log directory for this run.  Also used to
        # print some elapsed time and messages per second statistics.
        self.kernelWallClockStart = pd.Timestamp('now')
        # 4. 内核创建时间的初始化
        # self.kernelWallClockStart = pd.Timestamp('now')：将当前的时间戳（使用pandas的Timestamp对象表示）赋值给self.kernelWallClockStart。
        # 这个时间戳主要有两个用途：
        # 一是用于为每次运行创建一个唯一的日志目录，确保不同次运行的日志文件不会相互覆盖；
        # 二是用于计算模拟过程中的耗时和消息处理速度等统计信息。

        # TODO: This is financial, and so probably should not be here...
        self.meanResultByAgentType = {}
        self.agentCountByType = {}
        # 5. 代理类型统计信息的初始化
        # self.meanResultByAgentType = {}：初始化一个空字典meanResultByAgentType，用于存储每种代理类型的平均结果。
        # 从注释来看，这部分内容可能与金融模拟相关，后续可能会在模拟过程中记录不同类型代理的最终位置价值等信息，并计算平均值。
        # self.agentCountByType = {}：初始化一个空字典agentCountByType，用于存储每种代理类型的数量。
        # 在模拟过程中，可以通过这个字典统计不同类型代理的数量。

        # The Kernel maintains a summary log to which agents can write
        # information that should be centralized for very fast access
        # by separate statistical summary programs.  Detailed event
        # logging should go only to the agent's individual log.  This
        # is for things like "final position value" and such.
        self.summaryLog = []
        # 6. 汇总日志的初始化
        # self.summaryLog = []：初始化一个空列表summaryLog，用于存储汇总日志信息。
        # 内核会维护这个汇总日志，代理可以向其中写入一些需要集中存储的信息，比如最终位置价值等。
        # 这些信息可以被单独的统计汇总程序快速访问，而详细的事件日志则应该记录在每个代理的单独日志中。

        log_print("Kernel initialized: {}", self.name)
        # 7. 日志输出
        # log_print("Kernel initialized: {}", self.name)：调用自定义的log_print函数，
        # 输出一条日志信息，表示内核已经成功初始化，并显示内核的名称。

        ###############################################
        # self.prove_queue = queue.Queue()

    def __del__(self):
        self.dir_log_file.close()
    #  __del__ 是 Python 类中的一个特殊方法，也被称为析构函数。
    #  当一个对象的引用计数降为 0，即将被垃圾回收时，Python 解释器会自动调用该对象的 __del__ 方法。
    # 在这个 Kernel 类中，__del__ 方法的主要作用是关闭之前打开的日志文件，确保资源被正确释放，避免文件句柄泄漏。
    # self.dir_log_file 是在 runner 方法中打开的一个文件对象，用于将模拟过程中的日志信息写入到指定的文件中。
    # 在对象被销毁时，调用 close() 方法关闭该文件，释放系统资源。这样可以保证文件被正确关闭，避免数据丢失或文件损坏的问题。

    # This is called to actually start the simulation, once all agent
    # configuration is done.
    def runner(self, agents=[], manages: list = None,
               startTime=None, stopTime=None,
               num_simulations=1, defaultComputationDelay=1,
               defaultLatency=1, agentLatency=None, latencyNoise=[1.0],
               agentLatencyModel=None, skip_log=False,
               seed=None, oracle=None, log_dir=None, e_final_sum=False, d_final_sum=False):

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        self.agents = agents
        self.manages = [] if manages is None else manages
        self.e_final_sum = e_final_sum  # 编码数据
        self.d_final_sum = d_final_sum  # 解码数据

        self.dir_name = "data"
        os.makedirs(self.dir_name, exist_ok=True)

        self.dir_log_file = open(os.path.join(self.dir_name,
                                              f"{time.strftime('%Y_%m_%d %H_%M_%S', time.localtime())}.txt"),
                                 mode="w",
                                 encoding="utf-8")
        self.handle_T1_time = dict()  # 收到请求证明到生成pro_c的时间
        self.handle_T2_time = dict()
        self.handle_T3_time = dict()
        self.clients_pro_len = dict()
        self.clients_iter_numbers = 0

        # Simulation custom state in a freeform dictionary.  Allows config files
        # that drive multiple simulations, or require the ability to generate
        # special logs after simulation, to obtain needed output without special
        # case code in the Kernel.  Per-agent state should be handled using the
        # provided updateAgentState() method.
        self.custom_state = {}

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.startTime = startTime
        self.stopTime = stopTime

        # The global seed, NOT used for anything agent-related.
        self.seed = seed

        # Should the Kernel skip writing agent logs?
        self.skip_log = skip_log

        # The data oracle for this simulation, if needed.
        self.oracle = oracle

        # If a log directory was not specified, use the initial wallclock.
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = str(int(self.kernelWallClockStart.timestamp()))

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation startTime.
        self.agentCurrentTimes = [self.startTime] * len(agents)

        # agentComputationDelays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        # TODO: this might someday change to pd.Timedelta objects.
        self.agentComputationDelays = [defaultComputationDelay] * len(agents)

        # If an agentLatencyModel is defined, it will be used instead of
        # the older, non-model-based attributes.
        self.agentLatencyModel = agentLatencyModel

        # If an agentLatencyModel is NOT defined, the older parameters:
        # agentLatency (or defaultLatency) and latencyNoise should be specified.
        # These should be considered deprecated and will be removed in the future.

        # If agentLatency is not defined, define it using the defaultLatency.
        # This matrix defines the communication delay between every pair of
        # agents.
        if agentLatency is None:
            self.agentLatency = [[defaultLatency] * len(agents)] * len(agents)
        else:
            self.agentLatency = agentLatency

        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero.  By default there is no noise
        # (100% chance to add zero ns extra delay).  Format is a list with
        # list index = ns extra delay, value = probability of this delay.
        self.latencyNoise = latencyNoise

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receiveMessage, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receiveMessage calls.  It is useful for
        # staggering of sent messages.
        self.currentAgentAdditionalDelay = 0

        log_print("Kernel started: {}", self.name)
        log_print("Simulation started!")
        # 1. 初始化部分
        # 这部分代码主要完成了模拟所需的各种参数和状态的初始化工作。
        # 创建 data 目录用于存储日志文件，并以当前时间命名创建一个日志文件。
        # 初始化各种时间记录字典和计数器。
        # 根据传入的参数设置模拟的开始时间、结束时间、随机种子等。
        # 初始化代理的当前时间、计算延迟、通信延迟等。

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script.
        for sim in range(num_simulations):
            log_print("Starting sim {}", sim)

            # Event notification for kernel init (agents should not try to
            # communicate with other agents, as order is unknown).  Agents
            # should initialize any internal resources that may be needed
            # to communicate with other agents during agent.kernelStarting().
            # Kernel passes self-reference for agents to retain, so they can
            # communicate with the kernel in the future (as it does not have
            # an agentID).
            log_print("\n--- Agent.kernelInitializing() ---")
            for agent in self.agents:
                agent.kernelInitializing(self)

            for manage in self.manages:
                manage.kernelInitializing(self)

            # Event notification for kernel start (agents may set up
            # communications or references to other agents, as all agents
            # are guaranteed to exist now).  Agents should obtain references
            # to other agents they require for proper operation (exchanges,
            # brokers, subscription services...).  Note that we generally
            # don't (and shouldn't) permit agents to get direct references
            # to other agents (like the exchange) as they could then bypass
            # the Kernel, and therefore simulation "physics" to send messages
            # directly and instantly or to perform disallowed direct inspection
            # of the other agent's state.  Agents should instead obtain the
            # agent ID of other agents, and communicate with them only via
            # the Kernel.  Direct references to utility objects that are not
            # agents are acceptable (e.g. oracles).
            log_print("\n--- Agent.kernelStarting() ---")
            for agent in self.agents:
                agent.kernelStarting(self.startTime)

            # Set the kernel to its startTime.
            self.currentTime = self.startTime
            log_print("\n--- Kernel Clock started ---")
            log_print("Kernel.currentTime is now {}", self.currentTime)

            # Start processing the Event Queue.
            log_print("\n--- Kernel Event Queue begins ---")
            log_print("Kernel will start processing messages.  Queue length: {}", len(self.messages.queue))

            # Track starting wall clock time and total message count for stats at the end.
            eventQueueWallClockStart = pd.Timestamp('now')
            ttl_messages = 0
            # 2. 模拟循环
            # 使用 for 循环进行多次模拟，每次模拟开始时输出日志。
            # 调用代理和管理者的 kernelInitializing 方法，通知它们进行初始化操作。
            # 调用代理的 kernelStarting 方法，通知它们模拟开始。
            # 设置内核的当前时间为模拟开始时间。
            # 记录事件队列开始处理的时间和消息总数。

            # Process messages until there aren't any (at which point there never can
            # be again, because agents only "wake" in response to messages), or until
            # the kernel stop time is reached.
            while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
                # Get the next message in timestamp order (delivery time) and extract it.
                self.currentTime, event = self.messages.get()
                msg_recipient, msg_type, msg = event

                # Periodically print the simulation time and total messages, even if muted.
                if ttl_messages % 100000 == 0:
                    print("\n--- Simulation time: {}, messages processed: {}, wallclock elapsed: {} ---\n".format(
                        self.fmtTime(self.currentTime), ttl_messages, pd.Timestamp('now') - eventQueueWallClockStart))

                log_print("\n--- Kernel Event Queue pop ---")
                log_print("Kernel handling {} message for agent {} at time {}",
                          msg_type, msg_recipient, self.fmtTime(self.currentTime))

                ttl_messages += 1

                # In between messages, always reset the currentAgentAdditionalDelay.
                self.currentAgentAdditionalDelay = 0

                # Dispatch message to agent.
                if msg_type == MessageType.WAKEUP:

                    # Who requested this wakeup call?
                    agent = msg_recipient

                    # Test to see if the agent is already in the future.  If so,
                    # delay the wakeup until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Push the wakeup call back into the PQ with a new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        log_print("Agent in future: wakeup requeued for {}",
                                  self.fmtTime(self.agentCurrentTimes[agent]))
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agentCurrentTimes[agent] = self.currentTime

                    # Wake the agent.
                    agents[agent].wakeup(self.currentTime)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)

                    log_print("After wakeup return, agent {} delayed from {} to {}",
                              agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

                elif msg_type == MessageType.MESSAGE:

                    # Who is receiving this message?
                    agent = msg_recipient

                    # Test to see if the agent is already in the future.  If so,
                    # delay the message until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Push the message back into the PQ with a new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        log_print("Agent in future: message requeued for {}",
                                  self.fmtTime(self.agentCurrentTimes[agent]))
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agentCurrentTimes[agent] = self.currentTime

                    # Deliver the message.
                    agents[agent].receiveMessage(self.currentTime, msg)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)

                    log_print("After receiveMessage return, agent {} delayed from {} to {}",
                              agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

                else:
                    raise ValueError("Unknown message type found in queue",
                                     "currentTime:", self.currentTime,
                                     "messageType:", self.msg.type)
            # 3. 消息处理循环
            # 使用 while 循环处理消息队列中的消息，直到队列为空或达到模拟结束时间。
            # 从消息队列中取出下一个消息，根据消息类型（WAKEUP 或 MESSAGE）进行不同的处理。
            # 如果代理的当前时间大于全局时间，将消息重新放入队列，等待代理可以处理时再取出。
            # 调用代理的 wakeup 或 receiveMessage 方法处理消息，并更新代理的当前时间。

            if self.messages.empty():
                log_print("\n--- Kernel Event Queue empty ---")

            if self.currentTime and (self.currentTime > self.stopTime):
                log_print("\n--- Kernel Stop Time surpassed ---")

            # Record wall clock stop time and elapsed time for stats at the end.
            eventQueueWallClockStop = pd.Timestamp('now')

            eventQueueWallClockElapsed = eventQueueWallClockStop - eventQueueWallClockStart

            # Event notification for kernel end (agents may communicate with
            # other agents, as all agents are still guaranteed to exist).
            # Agents should not destroy resources they may need to respond
            # to final communications from other agents.
            log_print("\n--- Agent.kernelStopping() ---")
            for agent in agents:
                agent.kernelStopping()

            # Event notification for kernel termination (agents should not
            # attempt communication with other agents, as order of termination
            # is unknown).  Agents should clean up all used resources as the
            # simulation program may not actually terminate if num_simulations > 1.
            log_print("\n--- Agent.kernelTerminating() ---")
            for agent in agents:
                agent.kernelTerminating()

            print("Event Queue elapsed: {}, messages: {}, messages per second: {:0.1f}".format(
                eventQueueWallClockElapsed, ttl_messages,
                ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, 's')))))
            log_print("Ending sim {}", sim)
        # 4. 模拟结束处理
        # 如果消息队列为空或超过模拟结束时间，输出相应的日志信息。
        # 记录事件队列处理结束的时间，并计算处理耗时。
        # 调用代理的 kernelStopping 和 kernelTerminating 方法，通知它们模拟结束并进行清理操作。
        # 输出事件队列处理的统计信息，包括耗时、处理的消息总数和消息处理速度

        # The Kernel adds a handful of custom state results for all simulations,
        # which configurations may use, print, log, or discard.
        self.custom_state['kernel_event_queue_elapsed_wallclock'] = eventQueueWallClockElapsed
        self.custom_state['kernel_slowest_agent_finish_time'] = max(self.agentCurrentTimes)

        # Agents will request the Kernel to serialize their agent logs, usually
        # during kernelTerminating, but the Kernel must write out the summary
        # log itself.
        self.writeSummaryLog()

        # This should perhaps be elsewhere, as it is explicitly financial, but it
        # is convenient to have a quick summary of the results for now.
        print("Mean ending value by agent type:")
        for a in self.meanResultByAgentType:
            value = self.meanResultByAgentType[a]
            count = self.agentCountByType[a]
            print("{}: {:d}".format(a, int(round(value / count))))

        print("Simulation ending!")
        self.save_T2_T3_data()
        return self.custom_state
    # 5. 保存结果和日志
    # 将事件队列处理的耗时和最慢代理的结束时间保存到自定义状态字典中。
    # 调用 writeSummaryLog 方法写入汇总日志。
    # 输出每种代理类型的平均结束值。
    # 调用 save_T2_T3_data 方法保存 T2、T3 时间数据。
    # 返回自定义状态字典。

    def sendMessage(self, sender=None, recipient=None, msg=None, delay=0, tag=None):
        # Called by an agent to send a message to another agent.  The kernel
        # supplies its own currentTime (i.e. "now") to prevent possible
        # abuse by agents.  The kernel will handle computational delay penalties
        # and/or network latency.  The message must derive from the message.Message class.
        # The optional delay parameter represents an agent's request for ADDITIONAL
        # delay (beyond the Kernel's mandatory computation + latency delays) to represent
        # parallel pipeline processing delays (that should delay the transmission of messages
        # but do not make the agent "busy" and unable to respond to new messages).

        if sender is None:
            raise ValueError("sendMessage() called without valid sender ID",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)

        if recipient is None:
            raise ValueError("sendMessage() called without valid recipient ID",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)

        if msg is None:
            raise ValueError("sendMessage() called with message == None",
                             "sender:", sender, "recipient:", recipient,
                             "msg:", msg)
        # 1. 参数有效性检查
        # 这部分代码检查传入的 sender、recipient 和 msg 是否有效。
        # 如果其中任何一个为 None，则抛出 ValueError 异常，提示相应的错误信息。


        # Apply the agent's current computation delay to effectively "send" the message
        # at the END of the agent's current computation period when it is done "thinking".
        # NOTE: sending multiple messages on a single wake will transmit all at the same
        # time, at the end of computation.  To avoid this, use Agent.delay() to accumulate
        # a temporary delay (current cycle only) that will also stagger messages.

        # The optional pipeline delay parameter DOES push the send time forward, since it
        # represents "thinking" time before the message would be sent.  We don't use this
        # for much yet, but it could be important later.

        # This means message delay (before latency) is the agent's standard computation delay
        # PLUS any accumulated delay for this wake cycle PLUS any one-time requested delay
        # for this specific message only.
        sentTime = self.currentTime + pd.Timedelta(self.agentComputationDelays[sender] +
                                                   self.currentAgentAdditionalDelay + delay)

        # 2. 计算消息发送时间
        # sentTime 表示消息实际发送的时间。它是当前时间（self.currentTime）加上发送者的计算延迟（self.agentComputationDelays[sender]）、
        # 当前累积的额外延迟（self.currentAgentAdditionalDelay）以及可选的额外延迟（delay）。
        # 这样设计是为了模拟代理在完成 “思考”（计算）后才发送消息，并且可以处理并行管道处理延迟。

        # Apply communication delay per the agentLatencyModel, if defined, or the
        # agentLatency matrix [sender][recipient] otherwise.
        if self.agentLatencyModel is not None:
            latency = self.agentLatencyModel.get_latency(sender_id=sender, recipient_id=recipient)
            deliverAt = sentTime + pd.Timedelta(latency)

            # Log time-in-flight if tagged.
            if tag: self.custom_state[tag] = self.custom_state.get(tag, pd.Timedelta(0)) + pd.Timedelta(latency)

            log_print(
                "Kernel applied latency {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                latency, self.currentAgentAdditionalDelay, delay, self.agents[sender].name, self.agents[recipient].name,
                self.fmtTime(deliverAt))
        else:
            latency = self.agentLatency[sender][recipient]
            noise = self.random_state.choice(len(self.latencyNoise), 1, self.latencyNoise)[0]
            deliverAt = sentTime + pd.Timedelta(latency + noise)
            log_print(
                "Kernel applied latency {}, noise {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                latency, noise, self.currentAgentAdditionalDelay, delay, self.agents[sender].name,
                self.agents[recipient].name,
                self.fmtTime(deliverAt))
        # 3. 计算消息延迟和交付时间
        # 如果 self.agentLatencyModel 不为 None，则使用该模型来获取发送者和接收者之间的通信延迟（latency），并计算消息的交付时间（deliverAt）。
        # 如果 tag 不为 None，则记录消息的飞行时间。
        # 如果 self.agentLatencyModel 为 None，则使用 self.agentLatency 矩阵中的延迟值，并添加随机噪声（noise）来模拟更真实的网络环境，然后计算交付时间。

        # Finally drop the message in the queue with priority == delivery time.
        self.messages.put((deliverAt, (recipient, MessageType.MESSAGE, msg)))
        # 4. 将消息放入消息队列
        # 最后，将消息以交付时间为优先级放入消息队列（self.messages）中。
        # 这样可以确保消息按交付时间的先后顺序被处理。

        log_print("Sent time: {}, current time {}, computation delay {}", sentTime, self.currentTime,
                  self.agentComputationDelays[sender])
        log_print("Message queued: {}", msg)
        # 5. 记录日志
        # 记录消息的发送时间、当前时间和计算延迟，并记录消息已被放入队列。


    def setWakeup(self, sender=None, requestedTime=None):
        # Called by an agent to receive a "wakeup call" from the kernel
        # at some requested future time.  Defaults to the next possible
        # timestamp.  Wakeup time cannot be the current time or a past time.
        # Sender is required and should be the ID of the agent making the call.
        # The agent is responsible for maintaining any required state; the
        # kernel will not supply any parameters to the wakeup() call.

        if requestedTime is None:
            requestedTime = self.currentTime + pd.TimeDelta(1)

        if sender is None:
            raise ValueError("setWakeup() called without valid sender ID",
                             "sender:", sender, "requestedTime:", requestedTime)

        if self.currentTime and (requestedTime < self.currentTime):
            raise ValueError("setWakeup() called with requested time not in future",
                             "currentTime:", self.currentTime,
                             "requestedTime:", requestedTime)

        log_print("Kernel adding wakeup for agent {} at time {}",
                  sender, self.fmtTime(requestedTime))

        self.messages.put((requestedTime,
                           (sender, MessageType.WAKEUP, None)))
    # 功能：该方法用于让代理请求内核在指定的未来时间向其发送一个 “唤醒调用”（wakeup call）。
    # 参数：
    # sender：请求唤醒的代理的 ID，必须提供。
    # requestedTime：请求的唤醒时间，若未提供则默认为当前时间加 1 个时间单位。
    # 代码逻辑：
    # 若 requestedTime 未提供，将其设置为当前时间加 1 个时间单位。
    # 检查 sender 是否有效，若为 None 则抛出 ValueError 异常。
    # 检查 requestedTime 是否在未来，若不是则抛出 ValueError 异常。
    # 记录日志，表示内核正在为指定代理添加唤醒调用。
    # 将唤醒消息（包含请求时间、接收代理 ID 和消息类型）放入消息队列中。

    def getAgentComputeDelay(self, sender=None):
        # Allows an agent to query its current computation delay.
        return self.agentComputationDelays[sender]
    # 功能：允许代理查询其当前的计算延迟。
    # 参数：
    # sender：查询计算延迟的代理的 ID。
    # 代码逻辑：直接从 self.agentComputationDelays 列表中返回指定代理的计算延迟。

    def setAgentComputeDelay(self, sender=None, requestedDelay=None):
        # Called by an agent to update its computation delay.  This does
        # not initiate a global delay, nor an immediate delay for the
        # agent.  Rather it sets the new default delay for the calling
        # agent.  The delay will be applied upon every return from wakeup
        # or recvMsg.  Note that this delay IS applied to any messages
        # sent by the agent during the current wake cycle (simulating the
        # messages popping out at the end of its "thinking" time).

        # Also note that we DO permit a computation delay of zero, but this should
        # really only be used for special or massively parallel agents.

        # requestedDelay should be in whole nanoseconds.
        if not type(requestedDelay) is int:
            raise ValueError("Requested computation delay must be whole nanoseconds.",
                             "requestedDelay:", requestedDelay)

        # requestedDelay must be non-negative.
        if not requestedDelay >= 0:
            raise ValueError("Requested computation delay must be non-negative nanoseconds.",
                             "requestedDelay:", requestedDelay)

        self.agentComputationDelays[sender] = requestedDelay
    #     功能：允许代理更新其计算延迟。该延迟是该代理的新默认延迟，会在每次唤醒或接收消息后应用。
    # 参数：
    # sender：更新计算延迟的代理的 ID。
    # requestedDelay：请求的新计算延迟，必须为非负整数（表示纳秒）。
    # 代码逻辑：
    # 检查 requestedDelay 是否为整数，若不是则抛出 ValueError 异常。
    # 检查 requestedDelay 是否为非负数，若不是则抛出 ValueError 异常。
    # 将指定代理的计算延迟更新为 requestedDelay。

    def delayAgent(self, sender=None, additionalDelay=None):
        # Called by an agent to accumulate temporary delay for the current wake cycle.
        # This will apply the total delay (at time of sendMessage) to each message,
        # and will modify the agent's next available time slot.  These happen on top
        # of the agent's compute delay BUT DO NOT ALTER IT.  (i.e. effects are transient)
        # Mostly useful for staggering outbound messages.

        # additionalDelay should be in whole nanoseconds.
        if not type(additionalDelay) is int:
            raise ValueError("Additional delay must be whole nanoseconds.",
                             "additionalDelay:", additionalDelay)

        # additionalDelay must be non-negative.
        if not additionalDelay >= 0:
            raise ValueError("Additional delay must be non-negative nanoseconds.",
                             "additionalDelay:", additionalDelay)

        self.currentAgentAdditionalDelay += additionalDelay
    #     功能：允许代理在当前唤醒周期内累积临时延迟。该延迟会应用到每个发送的消息上，但不会改变代理的默认计算延迟。
    # 参数：
    # sender：请求临时延迟的代理的 ID。
    # additionalDelay：额外的临时延迟，必须为非负整数（表示纳秒）。
    # 代码逻辑：
    # 检查 additionalDelay 是否为整数，若不是则抛出 ValueError 异常。
    # 检查 additionalDelay 是否为非负数，若不是则抛出 ValueError 异常。
    # 将 additionalDelay 累加到 self.currentAgentAdditionalDelay 中。

    def findAgentByType(self, type=None):
        # Called to request an arbitrary agent ID that matches the class or base class
        # passed as "type".  For example, any ExchangeAgent, or any NasdaqExchangeAgent.
        # This method is rather expensive, so the results should be cached by the caller!

        for agent in self.agents:
            if isinstance(agent, type):
                return agent.id
    # 功能：根据传入的类或基类类型，查找任意一个匹配该类型的代理的 ID。
    # 参数：
    # type：要查找的代理的类或基类类型。
    # 代码逻辑：遍历所有代理，使用 isinstance 函数检查每个代理是否属于指定类型，若找到则返回其 ID。
    # 由于需要遍历所有代理，该方法的性能开销较大，建议调用者缓存结果。

    def writeLog(self, sender, dfLog, filename=None):
        # Called by any agent, usually at the very end of the simulation just before
        # kernel shutdown, to write to disk any log dataframe it has been accumulating
        # during simulation.  The format can be decided by the agent, although changes
        # will require a special tool to read and parse the logs.  The Kernel places
        # the log in a unique directory per run, with one filename per agent, also
        # decided by the Kernel using agent type, id, etc.

        # If there are too many agents, placing all these files in a directory might
        # be unfortunate.  Also if there are too many agents, or if the logs are too
        # large, memory could become an issue.  In this case, we might have to take
        # a speed hit to write logs incrementally.

        # If filename is not None, it will be used as the filename.  Otherwise,
        # the Kernel will construct a filename based on the name of the Agent
        # requesting log archival.

        if self.skip_log: return

        path = os.path.join(".", "log", self.log_dir)

        if filename:
            file = "{}.bz2".format(filename)
        else:
            file = "{}.bz2".format(self.agents[sender].name.replace(" ", ""))

        if not os.path.exists(path):
            os.makedirs(path)

        dfLog.to_pickle(os.path.join(path, file), compression='bz2')
    #     功能：用于代理在模拟结束前将其累积的日志数据帧（dfLog）写入磁盘。
    # 参数：
    # sender：请求写入日志的代理的 ID。
    # dfLog：要写入磁盘的日志数据帧。
    # filename：可选的文件名，若未提供则根据代理名称生成。
    # 代码逻辑：
    # 若 self.skip_log 为 True，则直接返回，不进行日志写入操作。
    # 构建日志文件的路径，若目录不存在则创建。
    # 根据 filename 参数生成文件名。
    # 使用 to_pickle 方法将日志数据帧以压缩（bz2）格式保存到指定文件中。

    def appendSummaryLog(self, sender, eventType, event):
        # We don't even include a timestamp, because this log is for one-time-only
        # summary reporting, like starting cash, or ending cash.
        self.summaryLog.append({'AgentID'      : sender,
                                'AgentStrategy': self.agents[sender].type,
                                'EventType'    : eventType, 'Event': event})
    # 功能：将代理的摘要日志信息添加到 self.summaryLog 列表中。该日志用于一次性的摘要报告，如初始现金或最终现金等信息。
    # 参数：
    # sender：发送摘要日志信息的代理的 ID。
    # eventType：事件类型。
    # event：事件内容。
    # 代码逻辑：将包含代理 ID、代理策略、事件类型和事件内容的字典添加到 self.summaryLog 列表中。

    def writeSummaryLog(self):
        path = os.path.join(".", "log", self.log_dir)
        file = "summary_log.bz2"

        if not os.path.exists(path):
            os.makedirs(path)

        dfLog = pd.DataFrame(self.summaryLog)

        dfLog.to_pickle(os.path.join(path, file), compression='bz2')
    #     功能：将 self.summaryLog 列表中的摘要日志信息转换为数据帧，并以压缩（bz2）格式保存到磁盘。
    # 代码逻辑：
    # 构建摘要日志文件的路径，若目录不存在则创建。
    # 将 self.summaryLog 列表转换为数据帧 dfLog。
    # 使用 to_pickle 方法将数据帧保存到指定文件中。

    def updateAgentState(self, agent_id, state):
        """ Called by an agent that wishes to replace its custom state in the dictionary
            the Kernel will return at the end of simulation.  Shared state must be set directly,
            and agents should coordinate that non-destructively.

            Note that it is never necessary to use this kernel state dictionary for an agent
            to remember information about itself, only to report it back to the config file.
        """

        if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
        self.custom_state['agent_state'][agent_id] = state
    #     功能：允许代理更新其在模拟结束时内核返回的自定义状态字典中的状态信息。
    # 参数：
    # agent_id：要更新状态的代理的 ID。
    # state：新的状态信息。
    # 代码逻辑：
    # 若 self.custom_state 中不存在 'agent_state' 键，则创建一个空字典。
    # 将指定代理的状态信息更新为 state。

    @staticmethod
    def fmtTime(simulationTime):
        # The Kernel class knows how to pretty-print time.  It is assumed simulationTime
        # is in nanoseconds since midnight.  Note this is a static method which can be
        # called either on the class or an instance.

        # Try just returning the pd.Timestamp now.
        return (simulationTime)

        ns = simulationTime
        hr = int(ns / (1000000000 * 60 * 60))
        ns -= (hr * 1000000000 * 60 * 60)
        m = int(ns / (1000000000 * 60))
        ns -= (m * 1000000000 * 60)
        s = int(ns / 1000000000)
        ns = int(ns - (s * 1000000000))

        return "{:02d}:{:02d}:{:02d}.{:09d}".format(hr, m, s, ns)

    ###############################################
    def findAgentsByType(self, type):
        agents = list()
        for agent in self.agents:
            if isinstance(agent, type):
                agents.append(agent)
        return agents

    def again_verify(self, line_clients: list) -> list:
        """伪代码

        1.管理者随机生成alpha
          km是管理者与每个客户端一一对应的k
          需要初始化出与客户端数同等的数量k
          初始化出来后得到了管理者自己的km
        2.每个管理者将这个一一对应的key发送到每个客户端中去
        3.客户端收到每个管理者发送的km,并计算出自己的kc,其中将每个管理者的km单独存储起来

        4.每个管理者将自己的km与alpha发送给每个客户端
        5.每个客户端收到所有管理者发送来的km与alpha后，生成大K与大alpha(这个是每个客户端独有的)

        6.每个客户端生成自己的pro_c=kc+大alpha*ver_n，并将其给到服务端

        7.服务端收到每个客户端的pro_c，将其相加成为PRO
          并将服务端生成的PRO与final_sum发送每个客户端
        8.每个客户端收到PRO与final_sum后，使用公式：PRO-大K-大alpha*ver_n
        """
        self.manager_dict = dict(map(lambda x: (x.id, x), self.manages))
        self.clients = self.findAgentsByType(Client)
        self.clients_dict = dict(map(lambda x: (x.id, x), self.clients))
        # 1.管理者随机生成alpha
        #   km是管理者与每个客户端一一对应的k
        #   需要初始化出与客户端数同等的数量k
        #   初始化出来后得到了管理者自己的km

        # 客户端发起请求
        c_public_keys = []
        for c_id in line_clients:
            self.handle_T1_time[c_id] = []
            start = time.time()
            self.clients_dict[c_id].verify_init()
            end = time.time()
            self.handle_T1_time[c_id].append(end - start)

            start = time.time()
            c_public_keys.append(self.clients_dict[c_id].send_public_to_manage())
            end = time.time()
            self.handle_T1_time[c_id].append(end - start)

        # 生成管理端共享密钥
        m_public_keys = []
        for manage in self.manages:
            m_public_keys += manage.verify_init(c_public_keys)
        # 生成客户端共享密钥
        for m_d in m_public_keys:
            start = time.time()
            self.clients_dict[m_d["c_id"]].generate_public(m_d)
            end = time.time()
            self.handle_T1_time[m_d["c_id"]].append(end - start)

        # 聚合所有管理端的km 发送加密big_K与alpha
        sipher_data = []
        for manage in self.manages:
            manage.count_km()
            sipher_data += manage.send_cipher_text(line_clients)
        # 解密管理端big_K与alpha
        for c_data in sipher_data:
            start = time.time()
            self.clients_dict[c_data["c_id"]].decrypt_big_k_alpha(c_data)
            end = time.time()
            self.handle_T1_time[c_data["c_id"]].append(end - start)
        # 3.客户端收到每个管理者发送的km,并计算出自己的kc、km与alpha,其中将每个管理者的km单独存储起来
        for c_id in line_clients:
            start = time.time()
            self.clients_dict[c_id].handle_km_alpha()
            end = time.time()
            self.handle_T1_time[c_id].append(end - start)

        # 5.每个客户端生成自己的pro_c=kc+大alpha*ver_n，并将其给到服务端
        self.clients_pro = list()
        for c_id in line_clients:
            start = time.time()
            self.clients_pro.append(self.clients_dict[c_id].count_pro_c())
            end = time.time()
            self.handle_T1_time[c_id].append(end - start)

        for c_id, _t in self.handle_T1_time.copy().items():
            self.handle_T1_time[c_id] = sum(self.handle_T1_time[c_id])

        return self.clients_pro
        # 6.服务端收到每个客户端的pro_c，将其相加成为PRO
        #   并将服务端生成的PRO与final_sum发送每个客户端
        # 7.每个客户端收到PRO与final_sum后，使用公式：PRO-大K-大alpha*ver_n

    def file_write(self, write_txt: str):
        self.dir_log_file.write(write_txt)
        self.dir_log_file.flush()

    def handle_log_time(self, handle_data: dict, _type: str):
        log_time = dict()
        sum_sum = 0
        _clients = []
        for i, (c_id, _t) in enumerate(handle_data.items()):
            i += 1
            sum_sum += _t
            _clients.append(c_id)
            if i % 10 == 0:
                log_time[i] = {
                    "num"    : sum_sum,
                    "clients": _clients.copy(),
                }

        log_time[len(handle_data)] = {
            "num"    : sum_sum,
            "clients": _clients.copy(),
        }
        write_txt = f"{_type}\n"
        if _type == "BYTE":
            for k, v in log_time.items():
                write_txt += f"客户端数：{k}\t字节数：{v['num']}\t客户端：{v['clients']}\n"
        else:
            for k, v in log_time.items():
                write_txt += f"客户端数：{k}\t时间：{v['num']}\t客户端：{v['clients']}\n"
        self.file_write(write_txt)

    def save_T2_T3_data(self):
        self.file_write(f"Number of rounds {self.iterations}\n")
        self.handle_log_time(self.handle_T1_time, "T1")
        self.handle_T1_time.clear()
        self.handle_log_time(self.handle_T2_time, "T2")
        self.handle_T2_time.clear()
        self.handle_log_time(self.handle_T3_time, "T3")
        self.handle_T3_time.clear()
        self.handle_log_time(self.clients_pro_len, "BYTE")
        self.clients_pro_len.clear()
        self.file_write(f"PRO_LEN：{self.PRO_len}\n")
        self.file_write(f"score：{self.SCORE}\n")
        self.file_write(f"loss rate：{1 - self.SCORE}\n")
        self.file_write(f"finished iteration：{self.finished_iteration}\n\n")


    ######原来的#####

    # def finish_score(self, score, PRO_len, iterations, finished_iteration):
    #     self.SCORE = score
    #     self.PRO_len = PRO_len
    #     self.iterations = iterations
    #     self.finished_iteration = finished_iteration
    ######原来的#####

    ######新增的#####
    def finish_score(self, MAE, PRO_len, iterations, finished_iteration):
        self.SCORE = MAE
        self.PRO_len = PRO_len
        self.iterations = iterations
        self.finished_iteration = finished_iteration
    ######新增的#####